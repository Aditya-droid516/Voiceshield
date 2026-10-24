'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { float32ToInt16LE, PCM_WORKLET_SOURCE } from '@/lib/pcm'
import {
  REALTIME_CALL_WS_URL,
  SYNTHETIC_ALERT,
  TARGET_SAMPLE_RATE,
  isSyntheticAlert,
  type RealtimeFrame,
} from '@/lib/voiceshield-config'

export type CallPhase = 'idle' | 'connecting' | 'configuring' | 'streaming' | 'error'

export type ThreatPoint = { live: number; smoothed: number; t: number }

export type HudState = {
  status: 'idle' | 'silence' | 'analyzed'
  alert: string
  liveScore: number
  smoothedScore: number
  windowSeconds: number | null
}

const HISTORY_LENGTH = 60
const CONFIG_TIMEOUT_MS = 3000

const INITIAL_HUD: HudState = {
  status: 'idle',
  alert: 'Standby',
  liveScore: 0,
  smoothedScore: 0,
  windowSeconds: null,
}

type AudioGraph = {
  ctx: AudioContext
  stream: MediaStream
  source: MediaStreamAudioSourceNode
  processor: AudioWorkletNode | ScriptProcessorNode
  sink: GainNode
  analyser: AnalyserNode
}

async function createAudioContext(stream: MediaStream) {
  try {
    const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE })
    const source = ctx.createMediaStreamSource(stream)
    return { ctx, source }
  } catch {
    // Some browsers (e.g. Firefox) refuse to resample a mic stream into a 16 kHz context.
    const ctx = new AudioContext()
    const source = ctx.createMediaStreamSource(stream)
    return { ctx, source }
  }
}

export function useRealtimeCall() {
  const [phase, setPhase] = useState<CallPhase>('idle')
  const [error, setError] = useState<string | null>(null)
  const [hud, setHud] = useState<HudState>(INITIAL_HUD)
  const [history, setHistory] = useState<ThreatPoint[]>([])
  const [sampleRate, setSampleRate] = useState<number | null>(null)
  const [framesSent, setFramesSent] = useState(0)
  const [demo, setDemo] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const graphRef = useRef<AudioGraph | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const configuredRef = useRef(false)
  const configTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const demoTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const sentCounterRef = useRef(0)
  const intentionalCloseRef = useRef(false)

  const applyFrame = useCallback((frame: RealtimeFrame) => {
    if (frame.status === 'silence') {
      setHud((prev) => ({ ...prev, status: 'silence', alert: frame.alert ?? 'Secure' }))
      setHistory((prev) => [...prev, { live: 0, smoothed: prev.at(-1)?.smoothed ?? 0, t: Date.now() }].slice(-HISTORY_LENGTH))
    } else if (frame.status === 'analyzed') {
      const live = Number(frame.live_threat_score) || 0
      const smoothed = Number(frame.smoothed_score) || 0
      setHud({
        status: 'analyzed',
        alert: frame.alert ?? 'Secure',
        liveScore: live,
        smoothedScore: smoothed,
        windowSeconds: frame.window_seconds ?? null,
      })
      setHistory((prev) => [...prev, { live, smoothed, t: Date.now() }].slice(-HISTORY_LENGTH))
    }
  }, [])

  const teardown = useCallback(() => {
    if (configTimerRef.current) clearTimeout(configTimerRef.current)
    if (demoTimerRef.current) clearInterval(demoTimerRef.current)
    configTimerRef.current = null
    demoTimerRef.current = null

    const graph = graphRef.current
    if (graph) {
      if ('port' in graph.processor) graph.processor.port.onmessage = null
      else graph.processor.onaudioprocess = null
      graph.source.disconnect()
      graph.processor.disconnect()
      graph.analyser.disconnect()
      graph.sink.disconnect()
      graph.stream.getTracks().forEach((track) => track.stop())
      graph.ctx.close().catch(() => {})
    }
    graphRef.current = null
    analyserRef.current = null

    const ws = wsRef.current
    if (ws) {
      intentionalCloseRef.current = true
      ws.onmessage = null
      ws.onerror = null
      ws.onclose = null
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close(1000, 'client stop')
    }
    wsRef.current = null
    configuredRef.current = false
  }, [])

  const stop = useCallback(() => {
    teardown()
    setPhase('idle')
    setDemo(false)
    setHud(INITIAL_HUD)
  }, [teardown])

  const fail = useCallback(
    (message: string) => {
      teardown()
      setError(message)
      setPhase('error')
      setDemo(false)
    },
    [teardown],
  )

  const sendPcm = useCallback((samples: Float32Array) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN || !configuredRef.current) return
    ws.send(float32ToInt16LE(samples))
    sentCounterRef.current += 1
    if (sentCounterRef.current % 8 === 0) setFramesSent(sentCounterRef.current)
  }, [])

  const start = useCallback(async () => {
    if (phase === 'connecting' || phase === 'configuring' || phase === 'streaming') return
    setError(null)
    setHistory([])
    setHud({ ...INITIAL_HUD, alert: 'Initializing' })
    setFramesSent(0)
    sentCounterRef.current = 0
    intentionalCloseRef.current = false
    setPhase('connecting')

    if (!navigator.mediaDevices?.getUserMedia) {
      fail('Microphone capture is not supported in this browser.')
      return
    }

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
    } catch (err) {
      fail(err instanceof Error && err.name === 'NotAllowedError' ? 'Microphone permission was denied.' : 'Unable to access the microphone.')
      return
    }

    try {
      const { ctx, source } = await createAudioContext(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 2048
      analyser.smoothingTimeConstant = 0.6
      const sink = ctx.createGain()
      sink.gain.value = 0

      let processor: AudioWorkletNode | ScriptProcessorNode
      if (ctx.audioWorklet) {
        const url = URL.createObjectURL(new Blob([PCM_WORKLET_SOURCE], { type: 'application/javascript' }))
        await ctx.audioWorklet.addModule(url)
        URL.revokeObjectURL(url)
        const node = new AudioWorkletNode(ctx, 'pcm-capture', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 })
        node.port.onmessage = (event: MessageEvent<Float32Array>) => sendPcm(event.data)
        processor = node
      } else {
        const node = ctx.createScriptProcessor(2048, 1, 1)
        node.onaudioprocess = (event) => sendPcm(new Float32Array(event.inputBuffer.getChannelData(0)))
        processor = node
      }

      source.connect(analyser)
      source.connect(processor)
      processor.connect(sink)
      sink.connect(ctx.destination)
      if (ctx.state === 'suspended') await ctx.resume()

      graphRef.current = { ctx, stream, source, processor, sink, analyser }
      analyserRef.current = analyser
      setSampleRate(ctx.sampleRate)

      const ws = new WebSocket(REALTIME_CALL_WS_URL)
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws

      ws.onopen = () => {
        setPhase('configuring')
        ws.send(JSON.stringify({ sample_rate: ctx.sampleRate }))
        configTimerRef.current = setTimeout(() => {
          if (!configuredRef.current) {
            configuredRef.current = true
            setPhase('streaming')
          }
        }, CONFIG_TIMEOUT_MS)
      }

      ws.onmessage = (event) => {
        if (typeof event.data !== 'string') return
        let frame: RealtimeFrame
        try {
          frame = JSON.parse(event.data)
        } catch {
          return
        }
        if (frame.status === 'configured') {
          if (configTimerRef.current) clearTimeout(configTimerRef.current)
          configuredRef.current = true
          setSampleRate(frame.sample_rate ?? ctx.sampleRate)
          setPhase('streaming')
          setHud((prev) => ({ ...prev, status: 'silence', alert: 'Secure' }))
          return
        }
        if (frame.status === 'error') {
          fail(frame.detail ?? frame.message ?? 'The server reported an error.')
          return
        }
        applyFrame(frame)
      }

      ws.onerror = () => {
        fail(`Could not reach ${REALTIME_CALL_WS_URL}. Make sure the VoiceShield backend is running.`)
      }

      ws.onclose = (event) => {
        if (intentionalCloseRef.current) return
        fail(event.reason ? `Connection closed: ${event.reason}` : 'Connection to the VoiceShield backend was closed.')
      }
    } catch {
      stream.getTracks().forEach((track) => track.stop())
      fail('Failed to initialize the audio pipeline.')
    }
  }, [phase, fail, sendPcm, applyFrame])

  const startDemo = useCallback(() => {
    teardown()
    setError(null)
    setHistory([])
    setDemo(true)
    setSampleRate(TARGET_SAMPLE_RATE)
    setPhase('streaming')
    setHud({ ...INITIAL_HUD, status: 'silence', alert: 'Secure' })

    let tick = 0
    let smoothed = 0
    let hotStreak = 0
    demoTimerRef.current = setInterval(() => {
      tick += 1
      const cycle = tick % 48
      if (cycle < 6) {
        applyFrame({ status: 'silence', alert: 'Secure' })
        smoothed *= 0.6
        hotStreak = 0
        return
      }
      const attack = cycle > 22 && cycle < 38
      const target = attack ? 0.88 + Math.random() * 0.1 : 0.12 + Math.random() * 0.22
      const live = Math.min(1, Math.max(0, target + (Math.random() - 0.5) * 0.15))
      smoothed = smoothed * 0.65 + live * 0.35
      hotStreak = smoothed > 0.75 ? hotStreak + 1 : 0
      applyFrame({
        status: 'analyzed',
        live_threat_score: live,
        smoothed_score: smoothed,
        window_seconds: 2,
        alert: hotStreak >= 2 ? SYNTHETIC_ALERT : smoothed > 0.5 ? 'Elevated Risk' : 'Secure',
      })
    }, 500)
  }, [teardown, applyFrame])

  useEffect(() => teardown, [teardown])

  return {
    phase,
    error,
    hud,
    history,
    sampleRate,
    framesSent,
    demo,
    analyserRef,
    isThreat: isSyntheticAlert(hud.alert),
    isActive: phase === 'connecting' || phase === 'configuring' || phase === 'streaming',
    start,
    stop,
    startDemo,
    wsUrl: REALTIME_CALL_WS_URL,
  }
}
