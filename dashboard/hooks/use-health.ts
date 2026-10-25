'use client'

import useSWR from 'swr'
import { API_BASE_URL } from '@/lib/voiceshield-config'

export type HealthResponse = { status: string; model_loaded?: boolean; [key: string]: unknown }
export type HealthState = 'checking' | 'online' | 'degraded' | 'offline'

async function fetchHealth(url: string): Promise<HealthResponse> {
  const res = await fetch(url, { cache: 'no-store' })
  if (!res.ok) throw new Error(`Health check failed (${res.status})`)
  return res.json()
}

export function useHealth() {
  const { data, error, isLoading, mutate, isValidating } = useSWR(`${API_BASE_URL}/health`, fetchHealth, {
    refreshInterval: 15000,
    revalidateOnFocus: true,
    shouldRetryOnError: false,
  })

  let state: HealthState = 'checking'
  if (error) state = 'offline'
  else if (data) state = data.status === 'ok' && data.model_loaded === true ? 'online' : 'degraded'
  else if (!isLoading) state = 'offline'

  return { state, data, refresh: () => mutate(), isValidating }
}
