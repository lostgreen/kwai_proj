import axios from 'axios'
import type { AnalysisResult, ExampleSummary } from '@/types/analysis'

const api = axios.create({
  baseURL: '/api',
})

export async function fetchExamples(): Promise<ExampleSummary[]> {
  const { data } = await api.get<ExampleSummary[]>('/examples')
  return data
}

export async function fetchExample(id: string): Promise<AnalysisResult> {
  const { data } = await api.get<AnalysisResult>(`/examples/${id}`)
  return data
}

export async function analyzeVideo(file: File): Promise<{ task_id: string }> {
  const form = new FormData()
  form.append('video', file)
  const { data } = await api.post<{ task_id: string }>('/analyze', form)
  return data
}
