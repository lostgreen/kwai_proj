<script setup lang="ts">
import { computed } from 'vue'
import type { Hierarchy } from '@/types/analysis'

const props = defineProps<{
  hierarchy: Hierarchy
  duration: number
  currentTime: number
  selectedId: string | null
}>()

const emit = defineEmits<{
  select: [id: string]
}>()

const layers = computed(() => [
  { label: 'L1', badge: '#6366f1', segments: props.hierarchy.L1_phases },
  { label: 'L2', badge: '#10b981', segments: props.hierarchy.L2_events },
  { label: 'L3', badge: '#f59e0b', segments: props.hierarchy.L3_actions },
])

function widthPct(start: number, end: number): string {
  return `${((end - start) / props.duration) * 100}%`
}

function isActive(start: number, end: number): boolean {
  return props.currentTime >= start && props.currentTime < end
}

const colors: Record<string, string[]> = {
  L1: ['#6366f1', '#818cf8', '#a5b4fc', '#6366f1', '#818cf8'],
  L2: ['#10b981', '#34d399', '#6ee7b7', '#10b981', '#34d399'],
  L3: ['#f59e0b', '#fbbf24', '#f59e0b', '#fbbf24', '#f59e0b'],
}
</script>

<template>
  <div class="timeline">
    <div v-for="layer in layers" :key="layer.label" class="layer">
      <div class="layer-label">
        <span class="badge" :style="{ background: layer.badge }">{{ layer.label }}</span>
      </div>
      <div class="bar">
        <div
          v-for="(seg, i) in layer.segments"
          :key="seg.id"
          class="segment"
          :class="{ active: isActive(seg.start, seg.end), selected: seg.id === selectedId }"
          :style="{
            width: widthPct(seg.start, seg.end),
            background: colors[layer.label]?.[i % 5] ?? '#888',
          }"
          :title="`${seg.label} (${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s)`"
          @click="emit('select', seg.id)"
        >
          <span class="seg-label">{{ seg.label }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.timeline { display: flex; flex-direction: column; gap: 12px; }
.layer { display: flex; flex-direction: column; gap: 4px; }
.layer-label { font-size: 12px; font-weight: 600; color: #666; display: flex; align-items: center; gap: 6px; }
.badge { font-size: 10px; padding: 2px 6px; border-radius: 10px; color: white; }
.bar { height: 36px; background: #f0f0f0; border-radius: 6px; display: flex; overflow: hidden; }
.segment {
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; color: white; cursor: pointer;
  border-right: 1px solid rgba(255,255,255,0.3);
  transition: opacity 0.2s;
  overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
}
.segment:hover { opacity: 0.85; }
.segment.active { box-shadow: inset 0 0 0 2px rgba(255,255,255,0.8); }
.segment.selected { box-shadow: inset 0 0 0 2px #fff, 0 0 0 2px #333; }
.seg-label { padding: 0 4px; overflow: hidden; text-overflow: ellipsis; }
</style>
