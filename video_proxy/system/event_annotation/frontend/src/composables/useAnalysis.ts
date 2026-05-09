import { ref, computed } from 'vue'
import type { AnalysisResult, ExampleSummary, Phase } from '@/types/analysis'
import { fetchExamples, fetchExample } from '@/api'

const examples = ref<ExampleSummary[]>([])
const currentResult = ref<AnalysisResult | null>(null)
const loading = ref(false)
const selectedSegmentId = ref<string | null>(null)

export function useAnalysis() {
  async function loadExamples() {
    examples.value = await fetchExamples()
  }

  async function selectExample(id: string) {
    loading.value = true
    try {
      currentResult.value = await fetchExample(id)
      selectedSegmentId.value = null
    } finally {
      loading.value = false
    }
  }

  function selectSegment(id: string | null) {
    selectedSegmentId.value = id
  }

  const selectedSegment = computed(() => {
    if (!currentResult.value || !selectedSegmentId.value) return null
    const id = selectedSegmentId.value
    const h = currentResult.value.hierarchy
    const phase = h.L1_phases.find(p => p.id === id)
    if (phase) return { level: 'L1' as const, data: phase }
    const event = h.L2_events.find(e => e.id === id)
    if (event) return { level: 'L2' as const, data: event }
    const action = h.L3_actions.find(a => a.id === id)
    if (action) return { level: 'L3' as const, data: action }
    return null
  })

  const childEvents = computed(() => {
    if (!currentResult.value || !selectedSegment.value) return []
    if (selectedSegment.value.level !== 'L1') return []
    const phase = selectedSegment.value.data as Phase
    return currentResult.value.hierarchy.L2_events.filter(
      e => phase.child_event_ids.includes(e.id)
    )
  })

  return {
    examples, currentResult, loading,
    selectedSegmentId, selectedSegment, childEvents,
    loadExamples, selectExample, selectSegment,
  }
}
