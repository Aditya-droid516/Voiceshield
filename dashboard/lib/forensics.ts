import { API_BASE_URL } from '@/lib/voiceshield-config'

export const ANALYZE_FILE_URL = `${API_BASE_URL}/api/v2/analyze-file`
export const ACCEPTED_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg'] as const
export const MAX_FILE_BYTES = 25 * 1024 * 1024

export type ChunkVerdict = 'Synthetic/Clone' | 'Authentic' | 'Silence' | string

export type TimelineChunk = {
  chunk_id: number
  time_window: string
  start_seconds: number
  end_seconds: number
  fake_probability: number | null
  verdict: ChunkVerdict
}

export type MetadataFinding = {
  source: 'tags' | 'raw_bytes' | 'wav_chunk' | string
  kind: 'editor' | 'converter' | 'pro_audio' | 'info' | string
  detail: string
}

export type TechnicalSpecs = {
  format?: string | null
  duration_seconds?: number | null
  sample_rate_hz?: number | null
  channels?: number | null
  bitrate_kbps?: number | null
  wav_chunks?: string[] | null
  wav_info_tags?: Record<string, string> | null
}

export type MetadataInspection = TechnicalSpecs & {
  potentially_edited?: boolean
  edit_confidence?: 'high' | 'medium' | 'low' | 'none' | string
  re_encoded_suspected?: boolean
  findings?: MetadataFinding[]
  technical_specs?: TechnicalSpecs
  forensic_note?: string
}

export type AnalyzeFileResponse = {
  original_filename?: string
  filename_assigned?: string
  size_bytes?: number
  file_info?: { original_filename?: string; filename_assigned?: string; size_bytes?: number }
  overall_verdict: string
  average_confidence_score?: number
  max_chunk_score?: number
  fake_chunk_ratio?: number
  timeline_analysis?: TimelineChunk[]
  metadata_inspection?: MetadataInspection
}

export function toPercent(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return 0
  return Math.max(0, Math.min(100, value <= 1 ? value * 100 : value))
}

export function formatBytes(bytes?: number) {
  if (bytes == null) return '—'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit++
  }
  return `${value.toFixed(value >= 100 ? 0 : 2)} ${units[unit]}`
}

export function validateAudioFile(file: File): string | null {
  const name = file.name.toLowerCase()
  if (!ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return `Unsupported format. Accepted: ${ACCEPTED_EXTENSIONS.join(', ')}`
  }
  if (file.size > MAX_FILE_BYTES) return `File is ${formatBytes(file.size)} — the limit is 25 MB.`
  if (file.size === 0) return 'File is empty.'
  return null
}

export type VerdictTone = 'crimson' | 'emerald' | 'slate'

export function verdictTone(verdict: string): VerdictTone {
  if (/deepfake|high risk|synthetic|clone/i.test(verdict)) return 'crimson'
  if (/no speech|silence/i.test(verdict)) return 'slate'
  return 'emerald'
}

export function errorDetail(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown })?.detail
  let message = ''
  if (typeof detail === 'string') message = detail
  else if (Array.isArray(detail)) message = detail.map((d) => (d as { msg?: string })?.msg ?? '').filter(Boolean).join('; ')
  const fallback: Record<number, string> = {
    400: 'Bad request — the audio could not be decoded.',
    413: 'Payload too large — files must be 25 MB or less.',
    422: 'Validation failed — send the audio as multipart field "file".',
  }
  return message || fallback[status] || `Request failed with status ${status}.`
}

export const SAMPLE_REPORT: AnalyzeFileResponse = {
  original_filename: 'ceo_voicemail_urgent.wav',
  filename_assigned: 'a3f9c1e2-7b4d-4e0a-9f12-ceo_voicemail_urgent.wav',
  size_bytes: 1_942_316,
  overall_verdict: 'High Risk: Deepfake Detected',
  average_confidence_score: 0.81,
  max_chunk_score: 0.97,
  fake_chunk_ratio: 0.67,
  timeline_analysis: [
    [0.12, 'Authentic'],
    [null, 'Silence'],
    [0.88, 'Synthetic/Clone'],
    [0.94, 'Synthetic/Clone'],
    [0.97, 'Synthetic/Clone'],
    [0.31, 'Authentic'],
    [0.79, 'Synthetic/Clone'],
    [0.91, 'Synthetic/Clone'],
    [null, 'Silence'],
    [0.86, 'Synthetic/Clone'],
  ].map(([p, verdict], i) => ({
    chunk_id: i,
    time_window: `${(i * 2).toFixed(1)}s - ${(i * 2 + 2).toFixed(1)}s`,
    start_seconds: i * 2,
    end_seconds: i * 2 + 2,
    fake_probability: p as number | null,
    verdict: verdict as string,
  })),
  metadata_inspection: {
    potentially_edited: true,
    edit_confidence: 'high',
    re_encoded_suspected: true,
    findings: [
      { source: 'tags', kind: 'editor', detail: 'Audacity 3.4.2' },
      { source: 'raw_bytes', kind: 'converter', detail: 'FFmpeg (Lavf60.16.100)' },
      { source: 'wav_chunk', kind: 'pro_audio', detail: 'Broadcast WAV (bext)' },
      { source: 'tags', kind: 'info', detail: 'ISFT: Adobe Audition' },
    ],
    technical_specs: {
      format: 'WAV (PCM_16)',
      duration_seconds: 20.04,
      sample_rate_hz: 48000,
      channels: 1,
      bitrate_kbps: 768,
      wav_chunks: ['fmt ', 'bext', 'LIST', 'data'],
      wav_info_tags: { ISFT: 'Adobe Audition', ICMT: 'export v2' },
    },
    forensic_note:
      'Metadata findings indicate processing history, not proof of manipulation. Editors and converters are commonly used on legitimate recordings.',
  },
}
