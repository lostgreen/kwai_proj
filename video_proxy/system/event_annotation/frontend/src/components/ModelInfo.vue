<script setup lang="ts">
defineProps<{
  modelName: string
  isMock: boolean
}>()
</script>

<template>
  <div class="model-info">
    <div class="info-row"><span class="label">Model</span><span>{{ modelName }}</span></div>
    <div class="info-row">
      <span class="label">Mode</span>
      <span class="mode-tag" :class="isMock ? 'mock' : 'live'">
        {{ isMock ? 'Pre-annotated' : 'Live Inference' }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.info-row { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 8px; }
.label { color: #888; }
.mode-tag { padding: 2px 8px; border-radius: 10px; font-size: 11px; }
.mode-tag.mock { background: #f0fdf4; color: #16a34a; }
.mode-tag.live { background: #eff6ff; color: #3b82f6; }
</style>
