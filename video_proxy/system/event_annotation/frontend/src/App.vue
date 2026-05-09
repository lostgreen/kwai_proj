<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import { NLayout, NLayoutHeader, NLayoutContent, NCard, NSpin } from 'naive-ui'
import { useAnalysis } from '@/composables/useAnalysis'
import { useVideoSync } from '@/composables/useVideoSync'
import ExampleSelector from '@/components/ExampleSelector.vue'
import VideoPlayer from '@/components/VideoPlayer.vue'
import HierarchicalTimeline from '@/components/HierarchicalTimeline.vue'
import CausalChain from '@/components/CausalChain.vue'
import CaptionPanel from '@/components/CaptionPanel.vue'
import SegmentDetail from '@/components/SegmentDetail.vue'
import ModelInfo from '@/components/ModelInfo.vue'

const {
  examples, currentResult, loading,
  selectedSegmentId, selectedSegment, childEvents,
  loadExamples, selectExample, selectSegment,
} = useAnalysis()

const { currentTime, setCurrentTime, findActiveSegments } = useVideoSync()

const videoPlayerRef = ref<InstanceType<typeof VideoPlayer>>()

const activeSegments = computed(() => findActiveSegments(currentResult.value))

function handleTimelineSelect(id: string) {
  selectSegment(id)
  if (!currentResult.value) return
  const h = currentResult.value.hierarchy
  const seg = [...h.L1_phases, ...h.L2_events, ...h.L3_actions].find(s => s.id === id)
  if (seg) videoPlayerRef.value?.seekTo(seg.start)
}

function handleCausalSelect(id: string) {
  handleTimelineSelect(id)
}

onMounted(() => {
  loadExamples()
})
</script>

<template>
  <NLayout style="height: 100vh;">
    <NLayoutHeader bordered style="padding: 12px 24px; background: #1a1a2e; color: white;">
      <h1 style="font-size: 18px; font-weight: 600; margin: 0;">VideoProxy — Video Understanding Visualization</h1>
    </NLayoutHeader>
    <NLayoutContent style="padding: 16px; overflow: auto;">
      <NSpin :show="loading">
        <div class="main-grid">
          <div class="left-panel">
            <NCard title="Video Analysis" size="small">
              <ExampleSelector
                :examples="examples"
                :current-id="currentResult?.video_id ?? null"
                @select="selectExample"
              />
              <VideoPlayer
                v-if="currentResult"
                ref="videoPlayerRef"
                :src="currentResult.video_url"
                style="margin-top: 12px;"
                @timeupdate="setCurrentTime"
              />
            </NCard>

            <NCard v-if="currentResult" title="Hierarchical Timeline" size="small" style="margin-top: 12px;">
              <HierarchicalTimeline
                :hierarchy="currentResult.hierarchy"
                :duration="currentResult.duration"
                :current-time="currentTime"
                :selected-id="selectedSegmentId"
                @select="handleTimelineSelect"
              />
            </NCard>

            <NCard v-if="currentResult && currentResult.causal_chain.length > 0" title="Event Causal Chain" size="small" style="margin-top: 12px;">
              <CausalChain
                :events="currentResult.hierarchy.L2_events"
                :chain="currentResult.causal_chain"
                :active-event-id="activeSegments.event?.id ?? null"
                @select="handleCausalSelect"
              />
            </NCard>
          </div>

          <div class="right-panel">
            <NCard v-if="currentResult" title="Video Caption" size="small">
              <CaptionPanel
                :caption="currentResult.caption"
                :domain="currentResult.domain"
                :duration="currentResult.duration"
              />
            </NCard>

            <NCard v-if="selectedSegment" title="Segment Detail" size="small" style="margin-top: 12px;">
              <SegmentDetail
                :level="selectedSegment.level"
                :segment="selectedSegment.data"
                :child-events="childEvents"
                @select-child="handleTimelineSelect"
              />
            </NCard>

            <NCard title="Model Info" size="small" style="margin-top: 12px;">
              <ModelInfo model-name="Qwen3-VL-4B-Instruct (RL fine-tuned)" :is-mock="true" />
            </NCard>
          </div>
        </div>
      </NSpin>
    </NLayoutContent>
  </NLayout>
</template>

<style>
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.main-grid { display: grid; grid-template-columns: 1fr 380px; gap: 16px; }
.left-panel { min-width: 0; }
.right-panel { min-width: 0; }
</style>
