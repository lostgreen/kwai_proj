import { ref } from 'vue'
import type { AnalysisResult } from '@/types/analysis'

const currentTime = ref(0)

export function useVideoSync() {
  function setCurrentTime(t: number) {
    currentTime.value = t
  }

  function findActiveSegments(result: AnalysisResult | null) {
    if (!result) return { phase: null, event: null, action: null }
    const t = currentTime.value
    const phase = result.hierarchy.L1_phases.find(p => t >= p.start && t < p.end) ?? null
    const event = result.hierarchy.L2_events.find(e => t >= e.start && t < e.end) ?? null
    const action = result.hierarchy.L3_actions.find(a => t >= a.start && t < a.end) ?? null
    return { phase, event, action }
  }

  return { currentTime, setCurrentTime, findActiveSegments }
}
