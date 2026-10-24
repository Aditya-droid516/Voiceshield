'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import {
  ANALYZE_FILE_URL,
  type AnalyzeFileResponse,
  SAMPLE_REPORT,
  errorDetail,
  validateAudioFile,
} from '@/lib/forensics'

export function useAnalyzeFile() {
  const [result, setResult] = useState<AnalyzeFileResponse | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [isSample, setIsSample] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl)
    }
  }, [audioUrl])

  useEffect(() => () => abortRef.current?.abort(), [])

  const analyze = useCallback(async (next: File) => {
    const invalid = validateAudioFile(next)
    if (invalid) {
      toast.error('File rejected', { description: invalid })
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setFile(next)
    setAudioUrl(URL.createObjectURL(next))
    setResult(null)
    setIsSample(false)
    setLoading(true)

    const form = new FormData()
    form.append('file', next)

    try {
      const res = await fetch(ANALYZE_FILE_URL, { method: 'POST', body: form, signal: controller.signal })
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        toast.error(`Error ${res.status}`, { description: errorDetail(res.status, body) })
        return
      }
      setResult(body as AnalyzeFileResponse)
      toast.success('Forensic scan complete', { description: (body as AnalyzeFileResponse).overall_verdict })
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      toast.error('Backend unreachable', {
        description: `Could not reach ${ANALYZE_FILE_URL}. Check that the API is running.`,
      })
    } finally {
      if (abortRef.current === controller) setLoading(false)
    }
  }, [])

  const loadSample = useCallback(() => {
    abortRef.current?.abort()
    setLoading(false)
    setFile(null)
    setAudioUrl(null)
    setIsSample(true)
    setResult(SAMPLE_REPORT)
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setLoading(false)
    setFile(null)
    setAudioUrl(null)
    setIsSample(false)
    setResult(null)
  }, [])

  return { result, file, audioUrl, loading, isSample, analyze, loadSample, reset }
}
