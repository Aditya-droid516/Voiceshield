'use client'

import { motion } from 'framer-motion'
import { Cpu, RefreshCw } from 'lucide-react'
import { type HealthState, useHealth } from '@/hooks/use-health'
import { cn } from '@/lib/utils'

const STATE_UI: Record<HealthState, { label: string; detail: string; className: string }> = {
  checking: {
    label: 'Checking',
    detail: 'Polling /health…',
    className: 'border-neon-cyan/40 bg-neon-cyan/10 text-neon-cyan',
  },
  online: {
    label: 'Online',
    detail: 'WavLM-base detector loaded',
    className:
      'border-neon-emerald/60 bg-neon-emerald/10 text-neon-emerald shadow-[0_0_20px_-4px_rgba(0,255,102,0.7)]',
  },
  degraded: {
    label: 'Degraded',
    detail: 'API up · model not loaded',
    className: 'border-neon-amber/60 bg-neon-amber/10 text-neon-amber shadow-[0_0_20px_-4px_rgba(255,184,0,0.7)]',
  },
  offline: {
    label: 'Offline',
    detail: 'Backend unreachable',
    className:
      'border-neon-crimson/60 bg-neon-crimson/10 text-neon-crimson shadow-[0_0_20px_-4px_rgba(255,0,85,0.7)]',
  },
}

export function HealthBadge() {
  const { state, refresh, isValidating } = useHealth()
  const ui = STATE_UI[state]

  return (
    <motion.button
      type="button"
      onClick={refresh}
      whileHover={{ scale: 1.06, y: -3 }}
      transition={{ type: 'spring', stiffness: 400, damping: 18 }}
      aria-label={`System health: ${ui.label}. ${ui.detail}. Click to re-check.`}
      aria-live="polite"
      className={cn(
        'group/health flex items-center gap-2.5 rounded-xl border px-3 py-1.5 outline-none transition-all duration-300 hover:brightness-125 focus-visible:ring-2 focus-visible:ring-neon-cyan/70',
        ui.className,
      )}
    >
      <span className="relative flex size-2.5">
        {state !== 'checking' && (
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-current opacity-60" />
        )}
        <span className="relative inline-flex size-2.5 rounded-full bg-current" />
      </span>
      <span className="flex flex-col items-start leading-tight">
        <span className="flex items-center gap-1.5 font-mono text-[11px] font-semibold uppercase tracking-[0.2em]">
          <Cpu className="size-3 transition-[filter] duration-300 group-hover/health:drop-shadow-[0_0_8px_currentColor]" aria-hidden="true" />
          {ui.label}
        </span>
        <span className="hidden text-[10px] text-muted-foreground lg:block">{ui.detail}</span>
      </span>
      <RefreshCw
        className={cn('size-3.5 opacity-60 transition-opacity group-hover/health:opacity-100', isValidating && 'animate-spin')}
        aria-hidden="true"
      />
    </motion.button>
  )
}
