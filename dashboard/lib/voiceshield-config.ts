export const API_BASE_URL = (process.env.NEXT_PUBLIC_VOICESHIELD_API_URL ?? 'http://localhost:8000').replace(
  /\/$/,
  '',
)

export const WS_BASE_URL = (
  process.env.NEXT_PUBLIC_VOICESHIELD_WS_URL ?? API_BASE_URL.replace(/^http/, 'ws')
).replace(/\/$/, '')

export const REALTIME_CALL_WS_URL = `${WS_BASE_URL}/ws/realtime-call`

export const TARGET_SAMPLE_RATE = 16000

export const SYNTHETIC_ALERT = '⚠️ WARNING: Synthetic Voice Signature Detected!'

export type RealtimeFrame =
  | { status: 'configured'; sample_rate: number }
  | { status: 'silence'; alert?: string }
  | {
      status: 'analyzed'
      live_threat_score: number
      smoothed_score: number
      window_seconds?: number
      alert?: string
    }
  | { status: 'error'; detail?: string; message?: string }

export function isSyntheticAlert(alert?: string) {
  if (!alert) return false
  return alert === SYNTHETIC_ALERT || /synthetic voice signature/i.test(alert)
}
