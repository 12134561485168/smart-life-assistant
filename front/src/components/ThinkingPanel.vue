<template>
  <div v-if="items.length" class="process-panel" :class="{ collapsed }">
    <div class="proc-title" @click="collapsed = !collapsed" title="折叠 / 展开思考过程">
      🧠 思考过程 <span class="proc-count">{{ items.length }}</span>
      <button v-if="!collapsed && items.length > 1" class="proc-toggle-all" @click.stop="toggleAllSteps">{{ openSteps.size ? '全部收起' : '全部展开' }}</button>
      <span class="p-chev">{{ collapsed ? '▸' : '▾' }}</span>
    </div>
    <div v-show="!collapsed" class="proc-steps">
      <!-- 按出现顺序组织：任务执行合并成一块，其余过程节点（输入处理/规划/汇总/反馈/检测等）逐条展示 -->
      <template v-for="(grp, gi) in groups" :key="gi">
        <!-- 任务执行块：只显示任务的问题与结果（result） -->
        <div v-if="grp.type === 'tool'" class="proc-step">
          <button class="p-head" @click="combinedOpen = !combinedOpen">
            <span class="dot done"></span>
            <span class="p-label">📋 任务执行（{{ grp.items.length }}）</span>
            <span class="p-status">完成</span>
            <span class="p-chev">{{ combinedOpen ? '▾' : '▸' }}</span>
          </button>
          <div v-show="combinedOpen" class="p-body">
            <div v-for="cs in combinedItems(grp.items)" :key="cs.id" class="tool-item">
              <div class="tool-q">▸ {{ cs.question }}</div>
              <div class="tool-text">{{ cs.content || '（无返回结果）' }}</div>
            </div>
          </div>
        </div>
        <!-- 其它过程节点：输入处理、规划、汇总、反馈、检测等逐条展示 -->
        <div v-else class="proc-step">
          <button class="p-head" @click="toggleStep(grp.item.id)">
            <span class="dot" :class="grp.item.running ? 'running' : (grp.item.status === 'failed' ? 'fail' : 'done')"></span>
            <span class="p-label">{{ iconFor(grp.item) }} {{ stepShortTitle(grp.item) }}</span>
            <span class="p-status" :class="grp.item.status === 'failed' ? 'fail' : ''">{{ statusText(grp.item) }}</span>
            <span class="p-chev">{{ isOpen(grp.item.id) ? '▾' : '▸' }}</span>
          </button>
          <div v-show="isOpen(grp.item.id)" class="p-body">
            <template v-if="grp.item.type === 'plan'">
              <!-- 计划：按执行顺序平铺展示任务（编号 1..n），不显示"第几步"分组 -->
              <template v-if="grp.item.steps && grp.item.steps.length">
                <div v-for="(s, si) in grp.item.steps" :key="s.id" class="plan-step">
                  <span class="plan-num">{{ si + 1 }}</span>
                  <span class="plan-q">{{ s.question }}</span>
                </div>
              </template>
              <div v-else class="info-text">正在为你的问题规划执行步骤…</div>
            </template>
            <template v-else-if="grp.item.type === 'note'">
              <!-- 补充信息提示：写在 AI 回答的思考过程中（不作为用户消息出现） -->
              <div class="tool-text note-text">{{ grp.item.content }}</div>
            </template>
            <template v-else>
              <div class="info-text">{{ infoText(grp.item) }}</div>
              <!-- 输入处理：额外展示清洗后的实际问题内容 -->
              <div v-if="grp.item.value && grp.item.value.content" class="tool-text process-content">{{ grp.item.value.content }}</div>
            </template>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

/* 思考面板为"各轮回答各自独立"的实例：折叠/展开、步骤展开状态均在本组件内自管理。
 * expand=true 时初始展开（生成中实时展示）；默认折叠为一行标题（回答完成后）。 */
const props = defineProps({
  items: { type: Array, default: () => [] },
  expand: { type: Boolean, default: false },
})

const collapsed = ref(!props.expand)
const combinedOpen = ref(!props.expand)
const openSteps = ref(new Set())

/* 把过程节点按出现顺序组织为展示分组：
 * - 相邻的"任务执行"（tool）合并成一块（只显示问题 + 结果 result）
 * - 其余节点（输入处理 / 规划 / 汇总 / 反馈 / 检测等）逐条展示，保留顺序 */
const groups = computed(() => {
  const out = []
  let cur = null
  for (const it of props.items) {
    if (it.type === 'tool') {
      if (!cur) { cur = { type: 'tool', items: [] }; out.push(cur) }
      cur.items.push(it)
    } else {
      cur = null
      out.push({ type: 'node', item: it })
    }
  }
  return out
})
/* 任务块内容：只取该任务的"问题"与"结果（result）"，不展示 key_data / 状态等后台字段。
 * 问题优先从各轮 plan 步骤里按任务 id 反查（允许多轮规划、跨轮同名任务），
 * 不再显示"子任务执行 · 步骤 tN"这类标题。 */
function combinedItems(toolGroup) {
  const allSteps = props.items
    .filter(i => i.type === 'plan' && Array.isArray(i.steps))
    .flatMap(i => i.steps)
  return (toolGroup || []).map(t => {
    const sid = String(String(t.id || '').split(':').pop())
    const step = allSteps.find(s => String(s.id) === sid)
    return {
      id: t.id,
      question: (step && step.question) || t.title || '',
      content: t.content || '',
    }
  })
}

function isOpen(id) { return openSteps.value.has(id) }
function toggleStep(id) {
  const s = openSteps.value
  if (s.has(id)) s.delete(id)
  else s.add(id)
  openSteps.value = new Set(s)
}
function toggleAllSteps() {
  openSteps.value = new Set(openSteps.value.size ? [] : props.items.map(p => p.id))
}

function iconFor(item) {
  const m = { input: '🚀', plan: '🗺️', schedule: '⚙️', run_subagent: '⚙️', weather: '☀️', food: '🍳', travel: '🚄', check: '✅', merge: '🧩', feedback: '🤝', guard: '🛡', fallback: '💬' }
  return m[item.node] || '•'
}
function statusText(item) {
  if (item.running) return '进行中'
  if (item.status === 'ok' || item.status === 'success') return '完成'
  if (item.status === 'failed' || item.status === 'error') return '失败'
  return '完成'
}
/* 标题已由后端带上执行波次与序号（如"第 2 步 · 步骤 4"），且不含内部节点类型，直接展示。 */
function stepShortTitle(item) {
  return item.title || item.label || ''
}
function infoText(item) {
  if (item.node === 'input') return item.value && item.value.message
  if (item.node === 'check') return '本批步骤执行完毕，进入下一步'
  if (item.node === 'merge') return '正在汇总各步骤结论…'
  if (item.node === 'feedback') {
    const v = item.value || {}
    if (!v.feedback_text) return '汇总结果已通过验收。'
    return `验收未通过：${v.feedback_text}`
  }
  if (item.node === 'guard') {
    if (item.value && item.value.passed === false) return '未通过内容安全/格式校验，已替换为兜底答复。'
    return '内容安全与格式校验均通过。'
  }
  if (item.node === 'fallback') return '未能识别任务，已给出兜底回复。'
  return '完成'
}

/* 计划步骤由后端按执行顺序（波次 + 任务编号）排好，前端直接按序编号平铺展示。 */
</script>