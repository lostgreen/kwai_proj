<script setup lang="ts">
import type { Phase, Event, Action } from '@/types/analysis'

defineProps<{
  level: 'L1' | 'L2' | 'L3'
  segment: Phase | Event | Action
  childEvents?: Event[]
}>()

const emit = defineEmits<{
  selectChild: [id: string]
}>()

function formatTime(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}
</script>

<template>
  <div class="detail">
    <div class="detail-section">
      <div class="detail-label">Selected Segment</div>
      <div class="detail-value">
        <span class="level-tag" :class="`tag-${level.toLowerCase()}`">{{ level }}</span>
        {{ segment.label }}
      </div>
      <div class="timestamp">{{ formatTime(segment.start) }} - {{ formatTime(segment.end) }} ({{ (segment.end - segment.start).toFixed(0) }}s)</div>
    </div>
    <div v-if="'description' in segment && segment.description" class="detail-section">
      <div class="detail-label">Description</div>
      <div class="detail-value">{{ segment.description }}</div>
    </div>
    <div v-if="childEvents && childEvents.length > 0" class="detail-section">
      <div class="detail-label">Child Events</div>
      <ul class="event-list">
        <li v-for="ev in childEvents" :key="ev.id" @click="emit('selectChild', ev.id)">
          <span class="dot" />
          {{ ev.label }}
          <span class="event-time">{{ formatTime(ev.start) }}-{{ formatTime(ev.end) }}</span>
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.detail-section { margin-bottom: 12px; }
.detail-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.detail-value { font-size: 13px; line-height: 1.5; }
.timestamp { color: #4f46e5; font-weight: 500; font-size: 13px; margin-top: 4px; }
.level-tag { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 11px; color: white; margin-right: 4px; }
.tag-l1 { background: #6366f1; }
.tag-l2 { background: #10b981; }
.tag-l3 { background: #f59e0b; }
.event-list { list-style: none; padding: 0; }
.event-list li { padding: 6px 8px; border-radius: 4px; font-size: 12px; display: flex; align-items: center; gap: 8px; cursor: pointer; }
.event-list li:hover { background: #f5f5f5; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; flex-shrink: 0; }
.event-time { color: #888; font-size: 11px; margin-left: auto; }
</style>
