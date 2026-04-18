<script setup lang="ts">
import type { CausalLink, Event } from '@/types/analysis'

const props = defineProps<{
  events: Event[]
  chain: CausalLink[]
  activeEventId: string | null
}>()

const emit = defineEmits<{
  select: [id: string]
}>()

function findEvent(id: string) {
  return props.events.find(e => e.id === id)
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}
</script>

<template>
  <div class="causal-chain">
    <template v-for="(link, i) in chain" :key="i">
      <div
        class="node"
        :class="{ active: findEvent(link.from_id)?.id === activeEventId }"
        @click="emit('select', link.from_id)"
      >
        <div class="node-label">{{ findEvent(link.from_id)?.label ?? link.from_id }}</div>
        <div class="node-time">{{ formatTime(findEvent(link.from_id)?.start ?? 0) }}-{{ formatTime(findEvent(link.from_id)?.end ?? 0) }}</div>
      </div>
      <div class="arrow" :title="link.relation">→</div>
    </template>
    <div
      v-if="chain.length > 0"
      class="node"
      :class="{ active: findEvent(chain[chain.length - 1].to_id)?.id === activeEventId }"
      @click="emit('select', chain[chain.length - 1].to_id)"
    >
      <div class="node-label">{{ findEvent(chain[chain.length - 1].to_id)?.label ?? chain[chain.length - 1].to_id }}</div>
      <div class="node-time">{{ formatTime(findEvent(chain[chain.length - 1].to_id)?.start ?? 0) }}-{{ formatTime(findEvent(chain[chain.length - 1].to_id)?.end ?? 0) }}</div>
    </div>
  </div>
</template>

<style scoped>
.causal-chain { display: flex; align-items: center; gap: 8px; overflow-x: auto; padding: 8px 0; }
.node {
  min-width: 100px; padding: 10px 14px; background: #f0fdf4; border: 1px solid #86efac;
  border-radius: 8px; text-align: center; flex-shrink: 0; cursor: pointer;
}
.node.active { background: #dcfce7; border-color: #22c55e; box-shadow: 0 0 0 2px rgba(34,197,94,0.2); }
.node-label { font-size: 12px; font-weight: 500; }
.node-time { font-size: 10px; color: #888; margin-top: 2px; }
.arrow { color: #059669; font-size: 18px; flex-shrink: 0; }
</style>
