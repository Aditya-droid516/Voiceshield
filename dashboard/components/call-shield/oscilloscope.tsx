'use client'

import { useEffect, useRef, type RefObject } from 'react'
import { useCanvasSize } from '@/hooks/use-canvas-size'

type OscilloscopeProps = {
  analyserRef: RefObject<AnalyserNode | null>
  color: string
  active: boolean
  synthetic: boolean
}

export function Oscilloscope({ analyserRef, color, active, synthetic }: OscilloscopeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const size = useCanvasSize(canvasRef)
  const propsRef = useRef({ color, active, synthetic })
  propsRef.current = { color, active, synthetic }

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx || !size.width) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = size.width * dpr
    canvas.height = size.height * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const buffer = new Float32Array(2048)
    let frame = 0
    let raf = 0

    const draw = () => {
      const { color, active, synthetic } = propsRef.current
      const { width: w, height: h } = size
      const mid = h / 2
      frame += 1
      ctx.clearRect(0, 0, w, h)

      ctx.strokeStyle = 'rgba(0,240,255,0.07)'
      ctx.lineWidth = 1
      for (let x = 0; x <= w; x += w / 12) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, h)
        ctx.stroke()
      }
      for (let y = 0; y <= h; y += h / 6) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(w, y)
        ctx.stroke()
      }

      const analyser = analyserRef.current
      const points = 256
      ctx.beginPath()
      if (analyser) {
        analyser.getFloatTimeDomainData(buffer)
        const step = Math.floor(analyser.fftSize / points)
        for (let i = 0; i < points; i++) {
          const v = buffer[i * step] ?? 0
          const x = (i / (points - 1)) * w
          const y = mid - Math.max(-1, Math.min(1, v * 2.2)) * (h * 0.45)
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
        }
      } else {
        const amp = synthetic ? h * 0.28 : active ? h * 0.04 : h * 0.015
        for (let i = 0; i < points; i++) {
          const t = i / (points - 1)
          const x = t * w
          const envelope = Math.sin(t * Math.PI)
          const noise = synthetic ? Math.sin(t * 90 + frame * 0.5) * 0.4 + (Math.random() - 0.5) * 0.5 : 0
          const v = (Math.sin(t * 18 + frame * 0.08) * 0.6 + noise) * envelope
          const y = mid - v * amp
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
        }
      }
      ctx.shadowColor = color
      ctx.shadowBlur = 16
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.shadowBlur = 0

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [size, analyserRef])

  return <canvas ref={canvasRef} className="block h-full w-full" aria-label="Live microphone waveform" role="img" />
}
