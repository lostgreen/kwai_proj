<script setup lang="ts">
defineProps<{
  caption: string
  domain?: string
  duration: number
}>()
</script>

<template>
  <div class="caption-panel">
    <div class="caption-box">{{ caption }}</div>
    <div class="tags">
      <span v-if="domain" class="tag tag-blue">{{ domain }}</span>
      <span class="tag tag-yellow">{{ duration.toFixed(0) }}s</span>
    </div>
  </div>
</template>

<style scoped>
.caption-box { background: #f8fafc; border-radius: 8px; padding: 12px; font-size: 13px; line-height: 1.6; border-left: 3px solid #4f46e5; }
.tags { margin-top: 8px; display: flex; gap: 4px; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; }
.tag-blue { background: #eff6ff; color: #3b82f6; }
.tag-yellow { background: #fffbeb; color: #d97706; }
</style>
