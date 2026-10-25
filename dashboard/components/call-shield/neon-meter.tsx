'use client'

import { motion } from 'framer-motion'
import type { LucideIcon } from 'lucide-react'
import { GlowIcon, type NeonTone } from '@/components/glow-icon'

const SEGMENTS = 24

function toneFor(value: number): { tone: NeonTone; hex: string; label: string } {
  if (value >= 0.75) return { tone: 'crimson', hex: '#FF0055', label: 'High Risk' }
  if (value >= 0.4) return { tone: 'amber', hex: '#FFB800', label: 'Medium' }
  return { tone: 'emerald', hex: '#00FF66', label: 'Low Risk' }
}

type NeonMeterProps = {
  label: string
  value: number
  icon: LucideIcon
  active: boolean
}

export function NeonMeter({ label, value, icon, active }: NeonMeterProps) {
  const clamped = Math.max(0, Math.min(1, value))
  const { tone, hex, label: riskLabel } = toneFor(clamped)
  const lit = active ? Math.round(clamped * SEGMENTS) : 0

  return (
    <motion.div
      whileHover={{ y: -6, scale: 1.02 }}
      transition={{ type: 'spring', stiffness: 360, damping: 22 }}
      className="glass group rounded-2xl p-4 transition-colors duration-300 hover:border-neon-cyan/70"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <GlowIcon icon={icon} label={label} tone={active ? tone : 'cyan'} size="sm" />
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">{label}</p>
            <p className="text-xs font-medium" style={{ color: active ? hex : undefined }}>
              {active ? riskLabel : 'Awaiting audio'}
            </p>
          </div>
        </div>
        <p
          className="font-mono text-3xl font-semibold tabular-nums"
          style={{ color: active ? hex : '#8b9bb4', textShadow: active ? `0 0 18px ${hex}` : undefined }}
        >
          {active ? clamped.toFixed(2) : '--'}
        </p>
      </div>

      <div
        className="mt-4 flex h-3 gap-[3px]"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={Number(clamped.toFixed(2))}
      >
        {Array.from({ length: SEGMENTS }, (_, i) => {
          const segHex = toneFor((i + 1) / SEGMENTS).hex
          const on = i < lit
          return (
            <span
              key={i}
              className="flex-1 rounded-[2px] transition-all duration-300"
              style={{
                background: on ? segHex : 'rgba(255,255,255,0.06)',
                boxShadow: on ? `0 0 10px ${segHex}` : undefined,
              }}
            />
          )
        })}
      </div>
    </motion.div>
  )
}
