<script setup lang="ts">
import type { ExampleSummary } from '@/types/analysis'

defineProps<{
  examples: ExampleSummary[]
  currentId: string | null
}>()

const emit = defineEmits<{
  select: [id: string]
}>()
</script>

<template>
  <div class="example-bar">
    <button
      v-for="ex in examples" :key="ex.id"
      class="example-btn"
      :class="{ active: ex.id === currentId }"
      @click="emit('select', ex.id)"
    >
      {{ ex.domain }} ({{ ex.duration.toFixed(0) }}s)
    </button>
  </div>
</template>

<style scoped>
.example-bar { display: flex; gap: 8px; flex-wrap: wrap; }
.example-btn { padding: 6px 14px; border: 1px solid #d0d0d0; border-radius: 6px; font-size: 12px; cursor: pointer; background: white; }
.example-btn.active { border-color: #4f46e5; background: #eff6ff; color: #4f46e5; }
</style>
