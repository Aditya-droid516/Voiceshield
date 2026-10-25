'use client'

import { useEffect, useRef } from 'react'
import type { ThreatPoint } from '@/hooks/use-realtime-call'
import { useCanvasSize } from '@/hooks/use-canvas-size'

const SLOTS = 60
const THRESHOLD = 0.75
const ROLL_MS = 480

type ThreatGraphProps = {
  history: ThreatPoint[]
  isThreat: boolean
}

export function ThreatGraph({ history, isThreat }: ThreatGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const size = useCanvasSize(canvasRef)
  const stateRef = useRef({ history, isThreat, updatedAt: 0 })

  useEffect(() => {
    stateRef.current = { history, isThreat, updatedAt: performance.now() }
  }, [history, isThreat])

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx || !size.width) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = size.width * dpr
    canvas.height = size.height * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    let raf = 0
    const pad = 6

    const draw = (now: number) => {
      const { history, isThreat, updatedAt } = stateRef.current
      const { width: w, height: h } = size
      const plotH = h - pad * 2
      const step = w / (SLOTS - 1)
      const progress = Math.min(1, (now - updatedAt) / ROLL_MS)
      const eased = 1 - Math.pow(1 - progress, 3)
      const n = history.length
      const xFor = (i: number) => w - (n - 1 - i) * step + (1 - eased) * step
      const yFor = (v: number) => pad + (1 - Math.max(0, Math.min(1, v))) * plotH

      ctx.clearRect(0, 0, w, h)

      ctx.strokeStyle = 'rgba(0,240,255,0.06)'
      ctx.lineWidth = 1
      for (let i = 0; i <= 4; i++) {
        const y = pad + (plotH / 4) * i
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(w, y)
        ctx.stroke()
      }

      const thresholdY = yFor(THRESHOLD)
      ctx.setLineDash([6, 6])
      ctx.strokeStyle = 'rgba(255,0,85,0.6)'
      ctx.beginPath()
      ctx.moveTo(0, thresholdY)
      ctx.lineTo(w, thresholdY)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = 'rgba(255,0,85,0.8)'
      ctx.font = '10px ui-monospace, monospace'
      ctx.fillText('0.75 THRESHOLD', 8, thresholdY - 6)

      if (n > 1) {
        const smoothedColor = isThreat ? '#FF0055' : '#00F0FF'

        const gradient = ctx.createLinearGradient(0, 0, 0, h)
        gradient.addColorStop(0, isThreat ? 'rgba(255,0,85,0.35)' : 'rgba(0,240,255,0.28)')
        gradient.addColorStop(1, 'rgba(0,0,0,0)')
        ctx.beginPath()
        ctx.moveTo(xFor(0), h)
        history.forEach((p, i) => ctx.lineTo(xFor(i), yFor(p.smoothed)))
        ctx.lineTo(xFor(n - 1), h)
        ctx.closePath()
        ctx.fillStyle = gradient
        ctx.fill()

        const line = (key: 'live' | 'smoothed', color: string, width: number) => {
          ctx.beginPath()
          history.forEach((p, i) => {
            const x = xFor(i)
            const y = yFor(p[key])
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
          })
          ctx.shadowColor = color
          ctx.shadowBlur = 14
          ctx.strokeStyle = color
          ctx.lineWidth = width
          ctx.lineJoin = 'round'
          ctx.stroke()
          ctx.shadowBlur = 0
        }
        line('live', 'rgba(176,38,255,0.9)', 1.5)
        line('smoothed', smoothedColor, 2.5)

        const last = history[n - 1]
        ctx.beginPath()
        ctx.arc(xFor(n - 1), yFor(last.smoothed), 4, 0, Math.PI * 2)
        ctx.fillStyle = smoothedColor
        ctx.shadowColor = smoothedColor
        ctx.shadowBlur = 18
        ctx.fill()
        ctx.shadowBlur = 0
      }

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [size])

  return <canvas ref={canvasRef} className="block h-full w-full" role="img" aria-label="Rolling threat score history" />
}
