<script setup>
import { ref, reactive, computed, onMounted, nextTick } from 'vue'
import { marked } from 'marked'
import ThinkingPanel from './components/ThinkingPanel.vue'

marked.setOptions({ gfm: true, breaks: true })

/* ============ 树模型状态 ============
 * 节点：{ id, role: 'root'|'human'|'ai', content, ts, checkpointId(仅AI), msg_id, parentId, children[] }
 * activePath：当前激活分支链（容器 → 叶子），UI 只渲染非容器节点
 */
const ROOT_ID = '__root__'
const treeState = reactive({ root: null })
const activePath = ref([])
const inputText = ref('')
const sending = ref(false)
const abortCtl = ref(null) // 当前进行中的流式请求控制器（「停止生成」时 abort）
const editing = ref(null) // 正在编辑的用户消息节点
/* 中断请教：check 缺数据时弹窗询问补充信息 */
const clarifyDialog = ref({ visible: false, question: '', cid: null, anchor: null })
const clarifyInput = ref('')

/* 流式过程：生成期间实时累积的步骤 */
const proc = ref([]) // ThinkingPanel 按各轮回答独立渲染

/* 分支内有正在生成的 AI 时，就地显示思考区，隐藏底部全局生成区 */
const hasAnyInlineGen = computed(() => activePath.value.some(n => n.role === 'ai' && n.generating))

/* ============ 用户登录 ============ */
const user = ref(localStorage.getItem('wx_user') || '')
const authToken = ref(localStorage.getItem('wx_token') || '')
const authed = computed(() => !!user.value && !!authToken.value)
const loginForm = reactive({ username: '', password: '' })
const loginMsg = ref('')

/* 统一附加服务端 Token；身份由 Token 决定 */
function authHeaders(extra = {}) {
  return authToken.value ? { ...extra, Authorization: `Bearer ${authToken.value}` } : extra
}
function handleUnauthorized(res) {
  if (res && res.status === 401) {
    loginMsg.value = '登录已过期，请重新登录'
    logout()
    return true
  }
  return false
}
const showSessions = ref(false)
const sessions = ref([]) // [{session_key, title, created_at, updated_at}]
const renamingKey = ref(null)
const renameText = ref('')

/* ============ 自定义弹窗（替代浏览器 alert/confirm） ============ */
const dialogState = ref({ visible: false, mode: 'alert', title: '', message: '', resolve: null })
function showAlert(message, title = '提示') {
  dialogState.value = { visible: true, mode: 'alert', title, message, resolve: null }
}
function showConfirm(message, title = '确认操作') {
  return new Promise(resolve => {
    dialogState.value = { visible: true, mode: 'confirm', title, message, resolve }
  })
}
function dialogClose(ok) {
  const d = dialogState.value
  const resolve = d.resolve
  dialogState.value = { visible: false, mode: 'alert', title: '', message: '', resolve: null }
  if (resolve) resolve(ok)
}

/* 当前会话密钥；null 表示新会话 */
const threadId = ref(localStorage.getItem('wx_current_session') || null)

function genId() {
  return 'wx_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}
function saveThread() {
  if (threadId.value) localStorage.setItem('wx_current_session', threadId.value)
}

/* 树工具：建节点 / 查找 / 路径 */
function makeRoot() {
  return { id: ROOT_ID, role: 'root', content: '', ts: 0, checkpointId: null, msg_id: null, parentId: null, children: [] }
}
function makeNode(role, content, parentId, checkpointId = null, msgId = null) {
  // 用户节点无 checkpointId；AI 节点存回答落盘点；两者都带后端消息 id
  return { id: genId(), role, content, ts: Date.now(), checkpointId, msg_id: msgId, parentId, children: [] }
}
/* 确保容器根存在；非 root 根节点包装进容器 */
function ensureRoot() {
  if (!treeState.root) {
    treeState.root = makeRoot()
  } else if (treeState.root.role !== 'root') {
    const old = treeState.root
    treeState.root = makeRoot()
    old.parentId = ROOT_ID
    treeState.root.children.push(old)
  }
}
function findNode(node, id) {
  if (!node) return null
  if (node.id === id) return node
  for (const c of node.children) {
    const r = findNode(c, id)
    if (r) return r
  }
  return null
}
/* 向上找最近一条 AI 的落盘点；无 AI 时返回 null */
function lastAICheckpoint(cur) {
  let n = cur
  while (n) {
    if (n.role === 'ai') return n.checkpointId ?? null
    n = n.parentId ? findNode(treeState.root, n.parentId) : null
  }
  return null
}
function pathTo(node) {
  const chain = []
  let cur = node
  while (cur) {
    chain.push(cur)
    cur = cur.parentId ? findNode(treeState.root, cur.parentId) : null
  }
  return chain.reverse()
}
function activateTo(node) {
  if (node && node.role === 'root') {
    if (node.children.length) return activateThrough(node.children[0])
    activePath.value = []
    return
  }
  activePath.value = node ? pathTo(node) : []
}
/* 切换到 child 所在分支并沿主线延伸到叶子 */
function activateThrough(child) {
  const path = pathTo(child)
  let cur = child
  while (cur.children.length) {
    cur = cur.children[0]
    path.push(cur)
  }
  activePath.value = path
}
function isDescendant(ancestor, node) {
  if (node.parentId == null) return false
  if (node.parentId === ancestor.id) return true
  const p = findNode(treeState.root, node.parentId)
  return p ? isDescendant(ancestor, p) : false
}

async function scrollToBottom() {
  await nextTick()
  const box = document.querySelector('.chat-body')
  if (box) box.scrollTop = box.scrollHeight
}

/* 从后端 /history 重建消息树：后端检查点为唯一权威来源 */
async function initFromHistory() {
  try {
    const res = await fetch(`/history?thread_id=${encodeURIComponent(threadId.value || '')}`, {
      headers: authHeaders(),
    })
    if (handleUnauthorized(res)) return
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || '加载历史失败')
    const msgs = data.messages || []
    if (!msgs.length) return
    ensureRoot()
    let prev = null
    let lastHuman = null
    // 问题索引：content -> 首个提问节点，用于归并回答分支
    const humanIndex = new Map()
    for (const m of msgs) {
      if (m.role === 'human') {
        const dup = humanIndex.get(m.content)
        if (dup) {
          // 同一问题再次出现：后续消息归到该问题下作为并列回答分支
          lastHuman = dup
          prev = dup
          continue
        }
        const node = makeNode('human', m.content, prev ? prev.id : ROOT_ID, null, m.id || null)
        if (!prev) treeState.root.children.push(node)
        else prev.children.push(node)
        humanIndex.set(m.content, node)
        lastHuman = node
        prev = node
      } else {
        // AI 回答挂到后端带回的 question 对应问题下，避免接到链尾
        const host = humanIndex.get(m.question) || lastHuman
        const node = makeNode('ai', m.content, host ? host.id : ROOT_ID, m.checkpoint_id || null, m.id || null)
        // 绑定该轮思考过程，供 ThinkingPanel 渲染
        if (Array.isArray(m.proc) && m.proc.length) node.proc = m.proc
        if (host) host.children.push(node)
        else treeState.root.children.push(node)
        lastHuman = host || lastHuman
        prev = node
      }
    }
    activateThrough(treeState.root.children[0])
    scrollToBottom()
  } catch (e) {
    console.error('加载历史失败', e)
  }
}

/* 核心请求：在 parentHuman 下以 SSE 流式追加 AI 回答节点。
 * 过程事件累积到 proc；打断时以最新 checkpoint 收尾。 */
async function ask(parentHuman, question, checkpointId, resume, immediateBranch = false) {
  sending.value = true
  proc.value = []
  const ctl = new AbortController()
  abortCtl.value = ctl
  let interruptCid = null // 打断时用于保存 AI 消息的 checkpoint
  // 重新回答：开始即建新分支，思考过程就地实时显示
  let host = null
  if (immediateBranch) {
    host = makeNode('ai', '', parentHuman.id, null, null)
    host.generating = true
    host.proc = proc.value // 与全局 proc 引用同一数组，实时同步思考过程
    parentHuman.children.push(host)
    activateTo(host)
    scrollToBottom()
  }
  try {
    const res = await fetch('/answer/stream', {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ thread_id: threadId.value, question, checkpoint_id: checkpointId, resume: resume || null }),
      signal: ctl.signal,
    })
    if (handleUnauthorized(res)) return
    if (!res.ok || !res.body) {
      let detail = res.statusText
      try { const d = await res.json(); detail = d.detail || detail } catch {}
      throw new Error(detail)
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buf = ''
    let result = null
    let afterCid = null
    let sessionFromBackend = null
    let humanMsgId = null
    let aiMsgId = null
    let clarified = null // 中断请教（check 缺必要信息 → interrupt）的澄清问题
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let idx
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        const ev = parseEvent(block)
        if (!ev) continue
        if (ev.event === 'process') {
          if (ev.data.checkpoint_id) interruptCid = ev.data.checkpoint_id
          upsertProc(ev.data)
        }
        else if (ev.event === 'session') {
          // 建连即回传会话密钥，避免首问被打断时会话丢失
          if (ev.data && ev.data.session_key && ev.data.session_key !== threadId.value) {
            threadId.value = ev.data.session_key
            saveThread()
            refreshSessions()
          }
        }
        else if (ev.event === 'interrupt') {
          // 缺必要信息：记录澄清问题与中断检查点，待补充后 resume
          clarified = ev.data.question || null
          if (ev.data.checkpoint_id) interruptCid = ev.data.checkpoint_id
          if (ev.data.session_key) {
            // 中断发生在会话密钥回传前时补登记
            threadId.value = ev.data.session_key
            saveThread()
            refreshSessions()
          }
        }
        else if (ev.event === 'done') {
          result = ev.data.result
          afterCid = ev.data.after_checkpoint_id
          sessionFromBackend = ev.data.session_key || null
          humanMsgId = ev.data.human_message_id || null
          aiMsgId = ev.data.ai_message_id || null
        }
        else if (ev.event === 'error') throw new Error((ev.data && ev.data.message) || '回答失败')
      }
    }

    // 中断请教：保留澄清 AI 节点，弹窗等待补充
    if (clarified) {
      let ai = host
      if (ai) {
        ai.content = clarified
        ai.checkpointId = interruptCid
        ai.generating = false // 思考过程已实时保存在 ai.proc
      } else {
        ai = makeNode('ai', clarified, parentHuman.id, interruptCid, null)
        ai.proc = proc.value.splice(0) // 思考过程保留在澄清节点上方
        parentHuman.children.push(ai)
        activateTo(ai)
      }
      clarifyDialog.value = { visible: true, question: clarified, cid: interruptCid, anchor: ai }
      clarifyInput.value = ''
      scrollToBottom()
      return
    }
    if (result === null) throw new Error('未收到回答结果')

    // 接收后端生成的会话密钥
    if (sessionFromBackend) {
      threadId.value = sessionFromBackend
      saveThread()
      refreshSessions()
    }
    let ai = host
    if (ai) {
      ai.content = result
      ai.checkpointId = afterCid
      ai.msg_id = aiMsgId || ai.msg_id
      ai.generating = false // 思考过程已实时保存在 ai.proc
    } else {
      ai = makeNode('ai', result, parentHuman.id, afterCid, aiMsgId)
      // 本轮思考过程挂到该 AI 回答上方
      ai.proc = proc.value.splice(0)
      // resume 时 parentHuman 是澄清节点，不能写入 human_message_id
      if (!resume) parentHuman.msg_id = humanMsgId || parentHuman.msg_id
      parentHuman.children.push(ai)
      activateTo(ai)
    }
    scrollToBottom()
  } catch (e) {
    if (e.name === 'AbortError') {
      // 已被废弃（abortCtl 已置空）：不再操作已重置的树
      if (abortCtl.value !== ctl) return
      // 用户点击「停止生成」：以最新 checkpoint 收尾 AI 消息（无检查点则只保留用户消息）
      if (interruptCid) {
        const note = buildInterruptedNote()
        if (host) {
          host.content = note
          host.checkpointId = interruptCid
          host.generating = false
        } else {
          const ai = makeNode('ai', note, parentHuman.id, interruptCid, null)
          ai.proc = proc.value.splice(0) // 打断后思考过程保留在该 AI 消息上方
          parentHuman.children.push(ai)
          activateTo(ai)
        }
        scrollToBottom()
      } else if (host) {
        // 无输出即被打断：撤销刚创建的空分支
        const idx = parentHuman.children.indexOf(host)
        if (idx >= 0) parentHuman.children.splice(idx, 1)
      } else {
        proc.value = []
      }
    } else {
      if (host) {
        const idx = parentHuman.children.indexOf(host)
        if (idx >= 0) parentHuman.children.splice(idx, 1)
      }
      showAlert('回答失败：' + e.message, '出错了')
      proc.value = []
    }
  } finally {
    if (abortCtl.value === ctl) abortCtl.value = null
    sending.value = false
  }
}

/* 打断收尾文案：保留已完成步骤摘要 */
function buildInterruptedNote() {
  const steps = proc.value.map(p => p.title || p.label || '').filter(Boolean)
  const suffix = steps.length ? `（已完成：${steps.join('、')}）` : '（尚未产出有效步骤）'
  return '⏹ 已停止生成。' + suffix + ' 可对此消息「🔄 重新生成」或继续追问。'
}

/* 打断：终止当前流式请求 */
function stopGeneration() {
  const ctl = abortCtl.value
  if (ctl) ctl.abort()
}

/* 废弃进行中的生成：先置空 abortCtl 再 abort，避免操作已重置的树 */
function abortActiveAsk() {
  const ctl = abortCtl.value
  if (ctl) {
    abortCtl.value = null
    ctl.abort()
  }
  sending.value = false
  proc.value = []
}

/* 解析一条 SSE 帧；不相关返回 null */
function parseEvent(block) {
  let evName = ''
  let dataStr = ''
  for (const line of block.split('\n')) {
    const t = line.trim()
    if (t.startsWith('event:')) evName = t.slice(6).trim()
    else if (t.startsWith('data:')) dataStr += t.slice(5).trim()
  }
  if (!dataStr) return null
  let data = null
  try { data = JSON.parse(dataStr) } catch { return null }
  if (evName === 'process' && data) return { event: 'process', data }
  if (evName === 'session' && data) return { event: 'session', data }
  if (evName === 'interrupt' && data) return { event: 'interrupt', data }
  if (evName === 'done' && data) return { event: 'done', data }
  if (evName === 'error' && data) return { event: 'error', data }
  return null
}

/* 累积 / 更新一条 process 事件并滚到底部 */
function upsertProc(d) {
  const i = proc.value.findIndex(x => x.id === d.id)
  if (i === -1) proc.value.push(Object.assign({ running: false }, d))
  else proc.value[i] = Object.assign({}, proc.value[i], d)
  scrollToBottom()
}

/* 提交补充信息，从中断检查点 resume 恢复执行 */
function submitClarify() {
  const q = clarifyInput.value.trim()
  const dlg = clarifyDialog.value
  if (!q || sending.value) return
  clarifyDialog.value = { visible: false, question: '', cid: null, anchor: null }
  clarifyInput.value = ''
  // 补充信息不生成用户气泡，从澄清节点下恢复执行
  ensureRoot()
  const parent = dlg.anchor || activePath.value[activePath.value.length - 1] || null
  ask(parent, q, dlg.cid, q) // 以中断检查点 + resume=补充信息恢复执行
}
/* 取消补充：关闭弹窗，保留澄清 AI 消息及其检查点 */
function cancelClarify() {
  clarifyDialog.value = { visible: false, question: '', cid: null, anchor: null }
  clarifyInput.value = ''
}

/* 发送输入：挂在当前激活路径末端继续提问 */
function submitQuestion() {
  const q = inputText.value.trim()
  if (!q || sending.value) return
  inputText.value = ''
  const parent = activePath.value[activePath.value.length - 1] ?? null
  const cid = lastAICheckpoint(parent)
  ensureRoot()
  const human = makeNode('human', q, parent ? parent.id : ROOT_ID)
  if (parent) parent.children.push(human)
  else treeState.root.children.push(human)
  activateTo(human)
  scrollToBottom()
  ask(human, q, cid)
}

/* 编辑用户消息：保存后在父节点下分裂新分支 */
function startEdit(node) {
  if (sending.value) return
  editing.value = node
  inputText.value = node.content
}
function cancelEdit() {
  editing.value = null
  inputText.value = ''
}
function submitEdit() {
  const q = inputText.value.trim()
  if (!q || sending.value) return
  const target = editing.value
  const parent = target.parentId && target.parentId !== ROOT_ID ? findNode(treeState.root, target.parentId) : null
  const cid = lastAICheckpoint(parent)
  ensureRoot()
  const human = makeNode('human', q, target.parentId && target.parentId !== ROOT_ID ? target.parentId : ROOT_ID)
  if (parent) parent.children.push(human)
  else treeState.root.children.push(human)
  activateTo(human)
  editing.value = null
  inputText.value = ''
  scrollToBottom()
  ask(human, q, cid)
}

/* 重新生成：从该轮提问前落盘点 fork，并立即切到新分支 */
function regenerate(node) {
  if (sending.value) return
  const parent = node.parentId ? findNode(treeState.root, node.parentId) : null
  const cid = lastAICheckpoint(parent)
  ask(parent, parent.content, cid, null, true)
}

/* 收集节点及子孙的后端消息 id */
function collectMsgIds(node, acc = []) {
  if (node.msg_id) acc.push(node.msg_id)
  for (const c of node.children || []) collectMsgIds(c, acc)
  return acc
}
/* 撤销：仅用户消息可撤销，并同步后端 */
async function revoke(node) {
  if (node.role === 'ai') return
  if (sending.value) return
  if (!await showConfirm('确定撤销这条用户消息及其下的所有分支吗？该操作会同步到服务端，历史记录中将不再显示。', '撤销确认')) return
  const ids = collectMsgIds(node)
  // 同步后端撤回，/history 重建时过滤
  if (threadId.value && ids.length) {
    fetch('/revoke', {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ thread_id: threadId.value, message_ids: ids }),
    }).catch(() => {})
  }
  if (node.parentId) {
    const parent = findNode(treeState.root, node.parentId)
    if (parent) {
      const idx = parent.children.findIndex(c => c.id === node.id)
      if (idx >= 0) parent.children.splice(idx, 1)
    }
  } else {
    treeState.root = null
  }
  if (activePath.value.some(n => n.id === node.id || isDescendant(node, n))) {
    const parent = node.parentId ? findNode(treeState.root, node.parentId) : null
    activateTo(parent)
    scrollToBottom()
  }
}

/* 分支可视化辅助 */
function hasBranches(node) {
  return node.children.length > 1
}
function isActiveBranch(child) {
  return activePath.value.some(n => n.id === child.id)
}
/* 去掉回答中残留的任务段标签 */
function cleanContent(text) {
  return (text || '').replace(/【\s*(weather|food)\s*】\s*[\r\n]*/g, '').trim()
}
/* 渲染 AI 回复的 markdown；先剥离 script 标签防注入 */
function renderMd(text) {
  if (!text) return ''
  const safe = String(text).replace(/<\s*script[\s\S]*?<\s*\/s*\s*script\s*>/gi, '')
  return marked.parse(safe) || ''
}
/* node 在当前激活分支中的序号（1 起）；未激活返回 0 */
function activeChildIndex(node) {
  const idx = (node.children || []).findIndex(ch => isActiveBranch(ch))
  return idx === -1 ? 0 : idx + 1
}
/* 会话起点多兄弟分支的切换状态；无则返回 null */
function startForkState() {
  const r = treeState.root
  if (!r || r.children.length < 2) return null
  return { idx: activeChildIndex(r), total: r.children.length }
}
/* 在子分支间切换（delta=-1 上一分支，1 下一分支） */
function switchBranch(node, delta) {
  const chs = node.children || []
  if (chs.length < 2) return
  let idx = chs.findIndex(ch => isActiveBranch(ch))
  if (idx === -1) idx = 0
  const next = Math.min(Math.max(idx + delta, 0), chs.length - 1)
  if (next !== idx) activateThrough(chs[next])
}
/* ============ 回答分支（重新生成的多个回答）============ */
/* 回答分支切换条显示在回答处；会话起点的多路线仍作为问题分支显示 */
function answerParent(node) {
  return node.parentId ? findNode(treeState.root, node.parentId) : null
}
function hasAnswerBranches(node) {
  const p = answerParent(node)
  return p && p.role === 'human' && (p.children || []).length > 1
}
function answerIndex(node) {
  const p = answerParent(node)
  const i = p ? (p.children || []).indexOf(node) : -1
  return i === -1 ? 1 : i + 1
}
function answerTotal(node) {
  const p = answerParent(node)
  return p ? (p.children || []).length : 1
}
function switchAnswerBranch(node, delta) {
  const p = answerParent(node)
  if (p) switchBranch(p, delta)
}
function fmt(ts) {
  const d = new Date(ts)
  const pad = (x) => String(x).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

/* ============ 登录 / 会话管理 ============ */
async function doLogin() {
  loginMsg.value = ''
  const name = loginForm.username.trim()
  if (!name || !loginForm.password) { loginMsg.value = '请输入账号和密码'; return }
  try {
    const res = await fetch('/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: name, password: loginForm.password }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || '登录失败')
    if (!data.token) throw new Error('登录响应缺少凭证，请检查后端配置')
    user.value = name
    authToken.value = data.token
    localStorage.setItem('wx_user', name)
    localStorage.setItem('wx_token', data.token)
    bootApp()
  } catch (e) {
    loginMsg.value = e.message
  }
}
function logout() {
  user.value = ''
  authToken.value = ''
  localStorage.removeItem('wx_user')
  localStorage.removeItem('wx_token')
  localStorage.removeItem('wx_current_session')
  abortActiveAsk()
  threadId.value = null
  treeState.root = null
  activePath.value = []
  editing.value = null
  loginForm.username = ''
  loginForm.password = ''
  loginMsg.value = ''
}

async function refreshSessions() {
  try {
    const res = await fetch('/sessions', { headers: authHeaders() })
    if (handleUnauthorized(res)) return
    const data = await res.json()
    sessions.value = data.sessions || []
  } catch (e) {
    console.error('加载会话列表失败', e)
  }
}
function switchSession(key) {
  showSessions.value = false
  if (key === threadId.value) return
  abortActiveAsk() // 停止当前会话的进行中生成，避免残留状态带入新会话
  threadId.value = key
  saveThread()
  treeState.root = null
  activePath.value = []
  editing.value = null
  inputText.value = ''
  // 历史会话不再依赖本地缓存：始终按会话 id 从后端 /history 重建消息树
  initFromHistory()
}
function startRename(sess) {
  renamingKey.value = sess.session_key
  renameText.value = sess.title
}
async function submitRename(sess) {
  const t = renameText.value.trim() || sess.title
  try {
    await fetch('/sessions/rename', {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ session_key: sess.session_key, title: t }),
    })
    sess.title = t
  } catch (e) {
    showAlert('重命名失败：' + e.message, '操作失败')
  }
  renamingKey.value = null
}
/* 删除会话：后端移除并更新本地列表 */
async function deleteSession(key) {
  const s = sessions.value.find(x => x.session_key === key)
  if (!s) return
  const name = s.title || '（未命名会话）'
  if (!await showConfirm(`确定删除会话「${name}」？删除后不可恢复。`, '删除会话')) return
  try {
    await fetch('/sessions/delete', {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ session_key: key }),
    })
  } catch (e) {
    showAlert('删除失败：' + e.message, '操作失败')
    return
  }
  sessions.value = sessions.value.filter(x => x.session_key !== key)
  // 若删除的是当前正在查看的会话，则回退为新的空白对话
  if (threadId.value === key) {
    abortActiveAsk()
    threadId.value = null
    localStorage.removeItem('wx_current_session')
    treeState.root = null
    activePath.value = []
    editing.value = null
    inputText.value = ''
    scrollToBottom()
  }
}
/* 新会话：仅重置为空白对话，密钥由后端生成 */
function newSession() {
  showSessions.value = false
  abortActiveAsk()
  threadId.value = null
  localStorage.removeItem('wx_current_session')
  treeState.root = null
  activePath.value = []
  editing.value = null
  inputText.value = ''
  refreshSessions()
}

/* 初始化：默认进入新的空白对话，历史会话从「我的会话」切换 */
function bootApp() {
  threadId.value = null
  localStorage.removeItem('wx_current_session')
  treeState.root = null
  activePath.value = []
  editing.value = null
  inputText.value = ''
  refreshSessions()
}

onMounted(() => {
  if (authed.value) bootApp()
})
</script>

<template>
  <!-- 登录界面 -->
  <div v-if="!authed" class="login-overlay">
    <div class="login-card">
      <div class="login-logo">☀️</div>
      <h2>万象 · 生活小助手</h2>
      <p class="login-sub">天气 · 美食 · 出行，愿陪你度过温暖日常</p>
      <form class="login-form" @submit.prevent="doLogin">
        <input v-model="loginForm.username" type="text" placeholder="账号" autocomplete="username" />
        <input v-model="loginForm.password" type="password" placeholder="密码" autocomplete="current-password" />
        <button class="btn primary" type="submit">登 录</button>
        <p v-if="loginMsg" class="login-msg">{{ loginMsg }}</p>
        <p class="login-tip">演示账号：admin / admin</p>
      </form>
    </div>
  </div>

  <!-- 主界面 -->
  <div v-else class="app">
    <header class="chat-header">
      <div class="brand">
        <span class="logo">☀️</span>
        <div>
          <h1>万象 · 生活小助手</h1>
          <p class="sub">天气 · 美食 · 出行，愿陪你度过温暖日常</p>
        </div>
      </div>
      <div class="header-actions">
        <button class="btn ghost" @click="showSessions = !showSessions">📚 我的会话</button>
        <button class="btn ghost" @click="newSession">🆕 新会话</button>
        <button class="btn ghost user-btn" title="退出登录" @click="logout">👤 {{ user }}</button>
      </div>
    </header>

    <!-- 会话侧边栏 -->
    <div v-if="showSessions" class="session-drawer" @click.self="showSessions = false">
      <div class="session-panel">
        <div class="session-head">
          <span>📚 我的会话（{{ sessions.length }}）</span>
          <button class="btn ghost" @click="showSessions = false">✕</button>
        </div>
        <button v-if="user" class="btn primary new-chat" @click="newSession">＋ 新建对话</button>
        <div class="session-list">
          <div v-if="!sessions.length" class="session-empty">暂无历史对话，开始一个新会话吧～</div>
          <div v-for="s in sessions" :key="s.session_key" class="session-item" :class="{ cur: s.session_key === threadId }">
            <div class="session-name" @click="switchSession(s.session_key)">
              <template v-if="renamingKey === s.session_key">
                <input
                  v-model="renameText"
                  class="rename-input"
                  :placeholder="s.title"
                  @keydown.enter.prevent="submitRename(s)"
                  @keydown.esc.prevent="renamingKey = null"
                  @click.stop
                />
              </template>
              <template v-else>
                <span class="session-title">{{ s.title || '（未命名会话）' }}</span>
              </template>
            </div>
            <div class="session-ops">
              <button class="mini-btn" title="重命名" @click.stop="startRename(s)">✏️</button>
              <button class="mini-btn danger" title="删除会话" @click.stop="deleteSession(s.session_key)">🗑</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <main class="chat-body">
      <div v-if="!activePath.length" class="empty">
        <div class="empty-icon">🌤️</div>
        <p>你好，我是「万象」～</p>
        <p class="empty-sub">可以问我今天要不要带伞、周末去哪里吃、出差坐高铁还是飞机。</p>
        <p class="empty-sub">支持对任意消息「编辑 / 重新生成」分裂新分支，可随时切换对话路径。</p>
      </div>

      <template v-for="(node, i) in activePath" :key="node.id">
        <template v-if="node.role !== 'root'">
          <!-- 该轮思考过程区；generating 时展开实时展示 -->
          <ThinkingPanel
            v-if="node.role === 'ai' && (node.generating || (node.proc && node.proc.length))"
            :items="node.proc || []"
            :expand="node.generating"
          />
          <div class="msg-row" :class="node.role === 'human' ? 'row-human' : 'row-ai'">
            <div v-if="node.role === 'ai'" class="avatar ai">万</div>
            <div class="bubble-wrap">
              <template v-if="node.role === 'ai'">
                <div v-if="node.generating" class="bubble bubble-ai typing"><span></span><span></span><span></span></div>
                <div v-else class="bubble bubble-ai" v-html="renderMd(node.content)"></div>
              </template>
              <!-- 回答分支切换条：同一问题下重新生成的多个回答 -->
              <div v-if="node.role === 'ai' && hasAnswerBranches(node)" class="branch-bar answer-branch">
                <button class="branch-nav" title="上一个回答" :disabled="answerIndex(node) <= 1" @click="switchAnswerBranch(node, -1)">‹</button>
                <span class="branch-count">&lt;{{ answerIndex(node) }}/{{ answerTotal(node) }}&gt;</span>
                <button class="branch-nav" title="下一个回答" :disabled="answerIndex(node) >= answerTotal(node)" @click="switchAnswerBranch(node, 1)">›</button>
              </div>
              <div v-if="node.role === 'human'" class="bubble bubble-human">{{ cleanContent(node.content) }}</div>
              <!-- 问题分支切换条：仅会话起点多路线时显示 -->
              <div v-if="node.role === 'human' && node.parentId === ROOT_ID && startForkState()" class="branch-bar">
                <button class="branch-nav" title="上一个问题分支（会话起点）" :disabled="startForkState().idx <= 1" @click="switchBranch(treeState.root, -1)">‹</button>
                <span class="branch-count">&lt;{{ startForkState().idx }}/{{ startForkState().total }}&gt;</span>
                <button class="branch-nav" title="下一个问题分支（会话起点）" :disabled="startForkState().idx >= startForkState().total" @click="switchBranch(treeState.root, 1)">›</button>
              </div>
              <div class="msg-meta">
                <span class="ts-tag">{{ fmt(node.ts) }}</span>
              </div>
              <div class="msg-actions">
                <button v-if="node.role === 'human'" :disabled="sending" @click="startEdit(node)">✎ 编辑</button>
                <button v-if="node.role === 'ai' && node.parentId" :disabled="sending" @click="regenerate(node)">
                  {{ sending ? '生成中…' : '🔄 重新生成' }}
                </button>
                <!-- AI 消息不允许撤销，仅用户消息可撤销 -->
                <button v-if="node.role === 'human'" :disabled="sending" @click="revoke(node)">🗑 撤销</button>
              </div>
            </div>
            <div v-if="node.role === 'human'" class="avatar human">我</div>
          </div>
        </template>
      </template>

      <!-- 全局思考过程：分支内已就地显示时不重复展示 -->
      <ThinkingPanel v-if="sending && proc.length && !hasAnyInlineGen" :items="proc" expand />

      <div v-if="sending && !hasAnyInlineGen" class="msg-row row-ai">
        <div class="avatar ai">万</div>
        <div class="bubble bubble-ai typing"><span></span><span></span><span></span></div>
      </div>
    </main>

    <footer class="chat-input">
      <div v-if="editing" class="editing-bar">
        <span>✎ 正在编辑：{{ (editing.content || '').slice(0, 28) }}{{ (editing.content || '').length > 28 ? '…' : '' }}</span>
        <button @click="cancelEdit">取消</button>
      </div>
      <div class="input-row">
        <textarea
          v-model="inputText"
          rows="1"
          :placeholder="sending
            ? '思考中…（可点击「停止生成」中断）'
            : editing
              ? '修改问题内容，保存后将分裂为新分支…'
              : '问天气、问菜谱……（Enter 发送，Shift+Enter 换行）'"
          @keydown.enter.exact.prevent="sending || !inputText.trim() ? null : editing ? submitEdit() : submitQuestion()"
        />
        <button
          class="btn primary"
          :class="{ stop: sending }"
          :disabled="!sending && !inputText.trim()"
          @click="sending ? stopGeneration() : editing ? submitEdit() : submitQuestion()"
        >
          {{ sending ? '⏹ 停止生成' : editing ? '保存为新分支' : '发送' }}
        </button>
      </div>
      <p class="hint">用户消息可「✎ 编辑 / 🗑 撤销」；同一位置多余回答用「‹ ›」切换对话路径；AI 回答只能「🔄 重新生成」，不可撤销</p>
    </footer>

    <!-- 自定义弹窗层（替代浏览器 alert/confirm） -->
    <div v-if="dialogState.visible" class="modal-mask" @click.self="dialogState.mode === 'alert' ? dialogClose(false) : null">
      <div class="modal-box">
        <h3 class="modal-title">{{ dialogState.title }}</h3>
        <p class="modal-msg">{{ dialogState.message }}</p>
        <div class="modal-actions">
          <button v-if="dialogState.mode === 'confirm'" class="btn ghost" @click="dialogClose(false)">取消</button>
          <button class="btn primary" @click="dialogClose(true)">{{ dialogState.mode === 'confirm' ? '确定' : '知道了' }}</button>
        </div>
      </div>
    </div>

    <!-- 中断请教弹窗：补充信息后 resume；取消则保留澄清消息 -->
    <div v-if="clarifyDialog.visible" class="modal-mask" @click.self="cancelClarify">
      <div class="modal-box">
        <h3 class="modal-title">❓ 我需要补充一些信息</h3>
        <p class="modal-msg">{{ clarifyDialog.question }}</p>
        <textarea
          v-model="clarifyInput"
          class="clarify-input"
          rows="3"
          placeholder="请补充上述问题所需的必要信息…（Enter 提交；不补充可选择「取消」）"
          @keydown.enter.exact.prevent="!clarifyInput.trim() || sending ? null : submitClarify()"
        ></textarea>
        <div class="modal-actions">
          <button class="btn ghost" @click="cancelClarify">取消</button>
          <button class="btn primary" :disabled="!clarifyInput.trim() || sending" @click="submitClarify">提交并继续回答</button>
        </div>
      </div>
    </div>
  </div>
</template>