<script setup>
import { ref, reactive, computed, onMounted, nextTick } from 'vue'
import ThinkingPanel from './components/ThinkingPanel.vue'

/* ============ 树模型状态 ============
 * treeState.root：容器根节点（role='root'，仅作根级用户的并列挂载点），每个节点：
 *   { id, role: 'root'|'human'|'ai', content, ts, checkpointId(仅AI), msg_id, parentId, children[] }
 * activePath：当前激活分支的节点链（容器 → 叶子），UI 只渲染非容器节点
 */
const ROOT_ID = '__root__'
const treeState = reactive({ root: null })
const activePath = ref([])
const inputText = ref('')
const sending = ref(false)
const abortCtl = ref(null) // 当前进行中的流式请求控制器（「停止生成」时 abort）
const editing = ref(null) // 正在编辑的用户消息节点
/* 中断请教：check 节点缺数据时以弹窗形式询问用户补充信息。
 * { visible, question: 澄清问题, cid: 中断检查点, anchor: 澄清问询 AI 节点 }
 * 用户提交补充 → resume 恢复执行；不补充选择取消 → 保留思考过程与检查点作为 AI 回答 */
const clarifyDialog = ref({ visible: false, question: '', cid: null, anchor: null })
const clarifyInput = ref('')

/* 流式过程：一次回答期间实时累积的节点步骤（可折叠） */
const proc = ref([]) // 生成中实时累积的步骤（思考面板由 ThinkingPanel 子组件按"各轮回答"独立渲染）

/* ============ 用户登录 ============ */
const user = ref(localStorage.getItem('wx_user') || '')
const authed = computed(() => !!user.value)
const loginForm = reactive({ username: '', password: '' })
const loginMsg = ref('')
const showSessions = ref(false)
const sessions = ref([]) // [{session_key, title, created_at, updated_at}]
const renamingKey = ref(null)
const renameText = ref('')

/* ============ 自定义弹窗（替代浏览器 alert/confirm） ============ */
const dialogState = ref({ visible: false, mode: 'alert', title: '', message: '', resolve: null })
/* 提示弹窗：仅"确定"按钮 */
function showAlert(message, title = '提示') {
  dialogState.value = { visible: true, mode: 'alert', title, message, resolve: null }
}
/* 确认弹窗：取消 / 确定，返回 Promise<boolean> */
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

/* 当前会话（会话密钥由后端在第一条用户消息时生成回传；新会话为 null） */
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
  // 用户节点不携带 checkpointId；AI 节点存储回答落盘点；两者都携带后端消息 id（撤销持久化）
  return { id: genId(), role, content, ts: Date.now(), checkpointId, msg_id: msgId, parentId, children: [] }
}
/* 确保容器根存在；兼容旧数据（根为 human 节点时包装进容器） */
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
/* 沿 cur 的父链向上找最近一条 AI 的落盘点；对话起始（无 AI）时返回 null */
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
/* 切换到 child 所在分支：根 → child，并沿该分支主线延伸到叶子 */
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

/* 从后端 /history 按会话 id 重建消息树：历史会话内容以后端检查点为唯一权威来源，
 * 不依赖本地缓存，选择任意历史会话都能在后端找到对应的消息树 */
async function initFromHistory() {
  try {
    const res = await fetch(`/history?thread_id=${encodeURIComponent(threadId.value || '')}&username=${encodeURIComponent(user.value)}`)
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || '加载历史失败')
    const msgs = data.messages || []
    if (!msgs.length) return
    ensureRoot()
    let prev = null
    for (const m of msgs) {
      const node = makeNode(m.role, m.content, prev ? prev.id : ROOT_ID, m.role === 'ai' ? m.checkpoint_id : null, m.id || null)
      // AI 节点绑定该轮"思考过程"（后端 /history 按轮次锚从 Redis 取回），供 ThinkingPanel 独立渲染
      if (m.role === 'ai' && Array.isArray(m.proc) && m.proc.length) node.proc = m.proc
      if (!prev) treeState.root.children.push(node)
      else prev.children.push(node)
      prev = node
    }
    activateThrough(treeState.root.children[0])
    scrollToBottom()
  } catch (e) {
    console.error('加载历史失败', e)
  }
}

/* 核心请求：在 parentHuman 下以 SSE 流式追加 AI 回答节点。
 * 过程节点（规划/工具/检查/反馈等）实时累积到 proc（可折叠），done 时产出最终结果作为 AI 消息。
 * 每条 process 事件携带当前 checkpoint_id（暂存到 interruptCid）；点击「停止生成」时，
 * 以打断前最后一次收到的 checkpoint 作为 AI 消息的 checkpoint 收尾，保留已进行的进度。 */
async function ask(parentHuman, question, checkpointId, resume) {
  sending.value = true
  proc.value = []
  const ctl = new AbortController()
  abortCtl.value = ctl
  let interruptCid = null // 打断时用于保存 AI 消息的 checkpoint（最新 process 事件携带）
  try {
    const res = await fetch('/answer/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: threadId.value, username: user.value, question, checkpoint_id: checkpointId, resume: resume || null }),
      signal: ctl.signal,
    })
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
          // 连接建立即回传会话密钥：新会话首问即使随即被打断也能先关联上会话，
          // 避免打断导致会话丢失（重复收到时幂等）
          if (ev.data && ev.data.session_key && ev.data.session_key !== threadId.value) {
            threadId.value = ev.data.session_key
            saveThread()
            refreshSessions()
          }
        }
        else if (ev.event === 'interrupt') {
          // check 发现缺必要信息：记录澄清问题与中断检查点，等待用户补充后 resume
          clarified = ev.data.question || null
          if (ev.data.checkpoint_id) interruptCid = ev.data.checkpoint_id
          if (ev.data.session_key) {
            // 中断发生在会话密钥回传前（新会话首问）时补登记，避免该会话未被前端关联
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

    // 中断请教：把澄清问题作为 AI 节点保留（思考过程挂在其上方），并弹窗询问用户补充。
    // 用户提交补充 → resume 恢复执行；不补充选择取消 → 保留该节点与检查点作为 AI 回答。
    if (clarified) {
      const ai = makeNode('ai', clarified, parentHuman.id, interruptCid, null)
      ai.proc = proc.value.splice(0) // 该轮思考过程保留在澄清节点上方
      parentHuman.children.push(ai)
      activateTo(ai)
      clarifyDialog.value = { visible: true, question: clarified, cid: interruptCid, anchor: ai }
      clarifyInput.value = ''
      scrollToBottom()
      return
    }
    if (result === null) throw new Error('未收到回答结果')

    // 新会话：接收后端生成的会话密钥并保存，之后继续该会话
    if (sessionFromBackend) {
      threadId.value = sessionFromBackend
      saveThread()
      refreshSessions()
    }
    const ai = makeNode('ai', result, parentHuman.id, afterCid, aiMsgId)
    // 本轮思考过程挂到该 AI 回答上方（整体折叠为一行标题，可展开查看），供后续保留
    ai.proc = proc.value.splice(0)
    // resume（补充信息恢复）时 parentHuman 是澄清问询的 AI 节点，不能把后端
    // 回传的 human_message_id 赋给它，否则撤销/重答会绑定到错误的用户消息
    if (!resume) parentHuman.msg_id = humanMsgId || parentHuman.msg_id
    parentHuman.children.push(ai)
    activateTo(ai)
    scrollToBottom()
  } catch (e) {
    if (e.name === 'AbortError') {
      // 已被切换会话/新会话/登出主动废弃（abortCtl 已被置空）：不再操作已废弃的树
      if (abortCtl.value !== ctl) return
      // 用户点击「停止生成」：以打断前的“当前 checkpoint”保存一条 AI 消息
      // （尚未产出任何检查点时说明刚开始即被打断，不补 AI 消息，仅停留在用户消息）
      if (interruptCid) {
        const note = buildInterruptedNote()
        const ai = makeNode('ai', note, parentHuman.id, interruptCid, null)
        ai.proc = proc.value.splice(0) // 打断后思考过程保留在该 AI 消息上方
        parentHuman.children.push(ai)
        activateTo(ai)
        scrollToBottom()
      } else {
        proc.value = []
      }
    } else {
      showAlert('回答失败：' + e.message, '出错了')
      proc.value = []
    }
  } finally {
    if (abortCtl.value === ctl) abortCtl.value = null
    sending.value = false
  }
}

/* 打断收尾文案：保留已完成步骤摘要，方便接着「重新生成 / 继续追问」 */
function buildInterruptedNote() {
  const steps = proc.value.map(p => p.title || p.label || '').filter(Boolean)
  const suffix = steps.length ? `（已完成：${steps.join('、')}）` : '（尚未产出有效步骤）'
  return '⏹ 已停止生成。' + suffix + ' 可对此消息「🔄 重新生成」或继续追问。'
}

/* 打断：终止当前流式请求（后端收到连接断开后即停止生成） */
function stopGeneration() {
  const ctl = abortCtl.value
  if (ctl) ctl.abort()
}

/* 主动废弃进行中的生成（切换会话/新会话/登出等）：先置空 abortCtl 再 abort，
 * 令 ask 的 AbortError 分支识别为“已废弃”而不去操作已被重置的树；同时复位发送态 */
function abortActiveAsk() {
  const ctl = abortCtl.value
  if (ctl) {
    abortCtl.value = null
    ctl.abort()
  }
  sending.value = false
  proc.value = []
}

/* 解析一条 SSE 帧（event:/data: 行）为 {event, data}；不相关的帧返回 null */
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

/* 把一条 process 事件累积/更新到 proc，并自动滚到底部 */
function upsertProc(d) {
  const i = proc.value.findIndex(x => x.id === d.id)
  if (i === -1) proc.value.push(Object.assign({ running: false }, d))
  else proc.value[i] = Object.assign({}, proc.value[i], d)
  scrollToBottom()
}

/* 中断请教弹窗：提交用户补充的信息，从中断检查点恢复执行（check 拿到后回 plan 重规划） */
function submitClarify() {
  const q = clarifyInput.value.trim()
  const dlg = clarifyDialog.value
  if (!q || sending.value) return
  clarifyDialog.value = { visible: false, question: '', cid: null, anchor: null }
  clarifyInput.value = ''
  // 补充信息不在前端生成用户消息气泡：从澄清问询 AI 节点下恢复执行，
  // 后端以"补充信息"提示节点写进本次回答（AI 消息）的思考过程。
  ensureRoot()
  const parent = dlg.anchor || activePath.value[activePath.value.length - 1] || null
  ask(parent, q, dlg.cid, q) // 以中断检查点 + resume=补充信息恢复执行
}
/* 中断请教弹窗：用户选择不补充（取消），关闭弹窗并保留澄清 AI 消息
 * （其上方思考过程与中断检查点已随该消息保留，作为本次 AI 回答内容） */
function cancelClarify() {
  clarifyDialog.value = { visible: false, question: '', cid: null, anchor: null }
  clarifyInput.value = ''
}

/* 输入框发送：挂在当前激活路径末端继续提问 */
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

/* 编辑用户消息：文本变更后分裂为新分支（同父节点下并列出新提问节点） */
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

/* 重新生成 AI 回答：以该轮提问前最近落盘点 fork 新分支 */
function regenerate(node) {
  if (sending.value) return
  const parent = node.parentId ? findNode(treeState.root, node.parentId) : null
  const cid = lastAICheckpoint(parent)
  ask(parent, parent.content, cid)
}

/* 收集某节点及其所有子孙的后端消息 id（撤销持久化） */
function collectMsgIds(node, acc = []) {
  if (node.msg_id) acc.push(node.msg_id)
  for (const c of node.children || []) collectMsgIds(c, acc)
  return acc
}
/* 撤销：仅用户消息可撤销；同时同步到后端，避免重新进入时错误渲染 */
async function revoke(node) {
  if (node.role === 'ai') return // AI 消息不允许撤销
  if (sending.value) return
  if (!await showConfirm('确定撤销这条用户消息及其下的所有分支吗？该操作会同步到服务端，历史记录中将不再显示。', '撤销确认')) return
  const ids = collectMsgIds(node)
  // 同步后端：标记这些消息为已撤销，/history 重建时不再渲染
  if (threadId.value && ids.length) {
    fetch('/revoke', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user.value, thread_id: threadId.value, message_ids: ids }),
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
/* 去掉回答中残留的任务段标签（如【weather】【food】），保证气温等内容正常显示 */
function cleanContent(text) {
  return (text || '').replace(/【\s*(weather|food)\s*】\s*[\r\n]*/g, '').trim()
}
/* 返回 node 在当前激活分支中的序号（1 起）；未激活返回 0 */
function activeChildIndex(node) {
  const idx = (node.children || []).findIndex(ch => isActiveBranch(ch))
  return idx === -1 ? 0 : idx + 1
}
/* 会话起点（第一条用户消息）存在多个兄弟分支时的切换状态（idx 1 起）；无则返回 null */
function startForkState() {
  const r = treeState.root
  if (!r || r.children.length < 2) return null
  return { idx: activeChildIndex(r), total: r.children.length }
}
/* 分支导航：在 node 的子分支间前后切换（delta=-1 上一分支，1 下一分支） */
function switchBranch(node, delta) {
  const chs = node.children || []
  if (chs.length < 2) return
  let idx = chs.findIndex(ch => isActiveBranch(ch))
  if (idx === -1) idx = 0
  const next = Math.min(Math.max(idx + delta, 0), chs.length - 1)
  if (next !== idx) activateThrough(chs[next])
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
    user.value = name
    localStorage.setItem('wx_user', name)
    bootApp()
  } catch (e) {
    loginMsg.value = e.message
  }
}
function logout() {
  user.value = ''
  localStorage.removeItem('wx_user')
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
    const res = await fetch(`/sessions?username=${encodeURIComponent(user.value)}`)
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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user.value, session_key: sess.session_key, title: t }),
    })
    sess.title = t
  } catch (e) {
    showAlert('重命名失败：' + e.message, '操作失败')
  }
  renamingKey.value = null
}
/* 删除会话：从后端移除并在本地列表消失（列表自此只保留未失效的会话） */
async function deleteSession(key) {
  const s = sessions.value.find(x => x.session_key === key)
  if (!s) return
  const name = s.title || '（未命名会话）'
  if (!await showConfirm(`确定删除会话「${name}」？删除后不可恢复。`, '删除会话')) return
  try {
    await fetch('/sessions/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user.value, session_key: key }),
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
/* 新会话：会话密钥由后端在第一条用户消息时生成，这里只重置为“新会话”状态 */
function newSession() {
  // 直接进入新对话，无需弹窗确认；当前会话仍保留在「我的会话」中可随时切回
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

/* 初始化：进入应用默认开启一个新的空白对话；历史会话可在「我的会话」中手动切换 */
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
          <!-- 该轮 AI 回答各自保留的思考过程区，显示在该回答上方（线性"先思考后回答"） -->
          <ThinkingPanel v-if="node.role === 'ai'" :items="node.proc" />
          <div class="msg-row" :class="node.role === 'human' ? 'row-human' : 'row-ai'">
            <div v-if="node.role === 'ai'" class="avatar ai">万</div>
            <div class="bubble-wrap">
              <div class="bubble" :class="node.role">{{ cleanContent(node.content) }}</div>
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

          <!-- 分支可视化与切换（中间用户消息/其下存在多子分支时）：统一展示方式 -->
          <div v-if="hasBranches(node)" class="branch-bar">
            <button class="branch-nav" title="上一个分支" :disabled="activeChildIndex(node) <= 1" @click="switchBranch(node, -1)">‹</button>
            <span class="branch-count">&lt;{{ activeChildIndex(node) }}/{{ node.children.length }}&gt;</span>
            <button class="branch-nav" title="下一个分支" :disabled="activeChildIndex(node) >= node.children.length" @click="switchBranch(node, 1)">›</button>
          </div>
          <!-- 会话起点（第一条用户消息）存在多个兄弟分支：切换栏同样放在对应消息下方 -->
          <div v-if="node.role === 'human' && node.parentId === ROOT_ID && startForkState()" class="branch-bar">
            <button class="branch-nav" title="上一个分支（会话起点）" :disabled="startForkState().idx <= 1" @click="switchBranch(treeState.root, -1)">‹</button>
            <span class="branch-count">&lt;{{ startForkState().idx }}/{{ startForkState().total }}&gt;</span>
            <button class="branch-nav" title="下一个分支（会话起点）" :disabled="startForkState().idx >= startForkState().total" @click="switchBranch(treeState.root, 1)">›</button>
          </div>
        </template>
      </template>

      <!-- 进行中的思考过程：紧贴 AI 回答出现位置（打字区上方），回答完成后转移为该回答上方的思考区 -->
      <ThinkingPanel v-if="sending && proc.length" :items="proc" expand />

      <div v-if="sending" class="msg-row row-ai">
        <div class="avatar ai">万</div>
        <div class="bubble ai typing"><span></span><span></span><span></span></div>
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

    <!-- 中断请教弹窗：check 缺数据时询问用户补充信息；不补充选「取消」则保留思考过程与检查点作为 AI 回答 -->
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