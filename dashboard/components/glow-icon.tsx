'use client'

import { motion } from 'framer-motion'
import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

const TONES = {
  cyan: 'text-neon-cyan hover:drop-shadow-[0_0_16px_rgba(0,240,255,0.95)] border-neon-cyan/30 hover:border-neon-cyan/80 hover:bg-neon-cyan/10',
  violet:
    'text-neon-violet hover:drop-shadow-[0_0_16px_rgba(176,38,255,0.95)] border-neon-violet/30 hover:border-neon-violet/80 hover:bg-neon-violet/10',
  emerald:
    'text-neon-emerald hover:drop-shadow-[0_0_16px_rgba(0,255,102,0.95)] border-neon-emerald/30 hover:border-neon-emerald/80 hover:bg-neon-emerald/10',
  amber:
    'text-neon-amber hover:drop-shadow-[0_0_16px_rgba(255,184,0,0.95)] border-neon-amber/30 hover:border-neon-amber/80 hover:bg-neon-amber/10',
  crimson:
    'text-neon-crimson hover:drop-shadow-[0_0_16px_rgba(255,0,85,0.95)] border-neon-crimson/30 hover:border-neon-crimson/80 hover:bg-neon-crimson/10',
} as const

export type NeonTone = keyof typeof TONES

type GlowIconProps = {
  icon: LucideIcon
  label: string
  tone?: NeonTone
  size?: 'sm' | 'md'
  className?: string
}

export function GlowIcon({ icon: Icon, label, tone = 'cyan', size = 'md', className }: GlowIconProps) {
  return (
    <motion.span
      role="img"
      aria-label={label}
      tabIndex={0}
      whileHover={{ scale: 1.15, y: -6 }}
      whileFocus={{ scale: 1.15, y: -6 }}
      transition={{ type: 'spring', stiffness: 420, damping: 18 }}
      className={cn(
        'group/glow relative inline-flex shrink-0 items-center justify-center rounded-lg border bg-white/5 outline-none transition-colors duration-300 focus-visible:ring-2 focus-visible:ring-neon-cyan/70',
        size === 'sm' ? 'size-8' : 'size-10',
        TONES[tone],
        className,
      )}
    >
      <Icon className={size === 'sm' ? 'size-4' : 'size-5'} aria-hidden="true" />
      <span
        aria-hidden="true"
        className="pointer-events-none absolute left-1/2 top-full z-50 mt-2.5 -translate-x-1/2 -translate-y-1 scale-90 whitespace-nowrap rounded-md border border-current/40 bg-void-2/90 px-2 py-1 font-mono text-[10px] uppercase tracking-widest opacity-0 shadow-[0_0_16px_currentColor] backdrop-blur-md transition-all duration-200 ease-out group-hover/glow:translate-y-0 group-hover/glow:scale-100 group-hover/glow:opacity-100 group-focus-visible/glow:translate-y-0 group-focus-visible/glow:scale-100 group-focus-visible/glow:opacity-100"
      >
        {label}
      </span>
    </motion.span>
  )
}
