'use client'

import { motion } from 'framer-motion'
import type { LucideIcon } from 'lucide-react'
import { GlowIcon, type NeonTone } from '@/components/glow-icon'

const TONE_HEX: Record<NeonTone, string> = {
  cyan: '#00F0FF',
  violet: '#B026FF',
  emerald: '#00FF66',
  amber: '#FFB800',
  crimson: '#FF0055',
}

export function riskTone(percent: number): NeonTone {
  if (percent >= 70) return 'crimson'
  if (percent >= 40) return 'amber'
  return 'emerald'
}

type RadialGaugeProps = { label: string; percent: number; icon: LucideIcon; caption: string }

export function RadialGauge({ label, percent, icon, caption }: RadialGaugeProps) {
  const tone = riskTone(percent)
  const color = TONE_HEX[tone]
  const radius = 52
  const circumference = 2 * Math.PI * radius

  return (
    <motion.figure
      whileHover={{ y: -6, scale: 1.04 }}
      transition={{ type: 'spring', stiffness: 380, damping: 20 }}
      className="glass flex flex-col items-center gap-3 rounded-2xl p-5 transition-colors duration-300 hover:border-neon-cyan/70"
    >
      <div className="flex w-full items-center justify-between gap-2">
        <figcaption className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">{label}</figcaption>
        <GlowIcon icon={icon} label={label} tone={tone} size="sm" />
      </div>
      <div className="relative size-36">
        <svg viewBox="0 0 120 120" className="size-full -rotate-90" role="img" aria-label={`${label}: ${percent.toFixed(1)}%`}>
          <circle cx="60" cy="60" r={radius} fill="none" stroke="rgb(255 255 255 / 0.06)" strokeWidth="9" />
          <motion.circle
            cx="60"
            cy="60"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="9"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: circumference * (1 - percent / 100) }}
            transition={{ duration: 1.2, ease: 'easeOut' }}
            style={{ filter: `drop-shadow(0 0 8px ${color})` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-3xl font-semibold tabular-nums" style={{ color, textShadow: `0 0 18px ${color}` }}>
            {percent.toFixed(1)}
            <span className="text-base">%</span>
          </span>
        </div>
      </div>
      <p className="text-center text-xs text-muted-foreground">{caption}</p>
    </motion.figure>
  )
}
