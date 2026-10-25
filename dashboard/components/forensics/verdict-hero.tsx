'use client'

import { motion } from 'framer-motion'
import { MicOff, ShieldAlert, ShieldCheck } from 'lucide-react'
import { GlowIcon } from '@/components/glow-icon'
import { verdictTone } from '@/lib/forensics'
import { cn } from '@/lib/utils'

const UI = {
  crimson: {
    icon: ShieldAlert,
    tone: 'crimson' as const,
    box: 'border-neon-crimson/70 bg-neon-crimson/10 threat-pulse',
    text: 'text-neon-crimson [text-shadow:0_0_28px_rgba(255,0,85,0.9)]',
    sub: 'Synthetic voice characteristics detected across multiple segments.',
  },
  emerald: {
    icon: ShieldCheck,
    tone: 'emerald' as const,
    box: 'border-neon-emerald/60 bg-neon-emerald/10 shadow-[0_0_60px_-12px_rgba(0,255,102,0.7)]',
    text: 'text-neon-emerald [text-shadow:0_0_28px_rgba(0,255,102,0.8)]',
    sub: 'No meaningful synthetic voice signature was found in this recording.',
  },
  slate: {
    icon: MicOff,
    tone: 'cyan' as const,
    box: 'border-slate-500/50 bg-slate-500/10',
    text: 'text-slate-300',
    sub: 'The detector found no voiced segments to evaluate.',
  },
}

export function VerdictHero({ verdict }: { verdict: string }) {
  const ui = UI[verdictTone(verdict)]
  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: 'spring', stiffness: 260, damping: 22 }}
      className={cn('flex flex-col items-start gap-4 rounded-2xl border-2 p-6 backdrop-blur-[20px] sm:flex-row sm:items-center sm:p-8', ui.box)}
      role="status"
    >
      <GlowIcon icon={ui.icon} label="Overall verdict" tone={ui.tone} />
      <div className="flex flex-col gap-1">
        <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-muted-foreground">Executive verdict</p>
        <p className={cn('text-balance text-3xl font-bold tracking-tight sm:text-5xl', ui.text)}>{verdict}</p>
        <p className="text-sm text-muted-foreground">{ui.sub}</p>
      </div>
    </motion.div>
  )
}
