export interface Phase {
  id: string
  label: string
  start: number
  end: number
  description: string
  child_event_ids: string[]
}

export interface Event {
  id: string
  label: string
  start: number
  end: number
  description: string
  parent_phase_id: string
  child_action_ids: string[]
}

export interface Action {
  id: string
  label: string
  start: number
  end: number
  parent_event_id: string
}

export interface CausalLink {
  from_id: string
  to_id: string
  relation: string
}

export interface KeyFrame {
  time: number
  url: string
}

export interface Hierarchy {
  L1_phases: Phase[]
  L2_events: Event[]
  L3_actions: Action[]
}

export interface AnalysisResult {
  video_id: string
  video_url: string
  duration: number
  caption: string
  domain?: string
  hierarchy: Hierarchy
  causal_chain: CausalLink[]
  key_frames: Record<string, KeyFrame[]>
}

export interface ExampleSummary {
  id: string
  title: string
  domain: string
  duration: number
  thumbnail_url: string
}
