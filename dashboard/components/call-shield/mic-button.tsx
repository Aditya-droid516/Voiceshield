'use client'

import { motion, useReducedMotion } from 'framer-motion'
import { Loader2, Mic, Square } from 'lucide-react'
import type { CallPhase } from '@/hooks/use-realtime-call'
import { cn } from '@/lib/utils'

type MicButtonProps = {
  phase: CallPhase
  isActive: boolean
  isThreat: boolean
  analyzing: boolean
  onToggle: () => void
}

export function MicButton({ phase, isActive, isThreat, analyzing, onToggle }: MicButtonProps) {
  const reduceMotion = useReducedMotion()
  const pending = phase === 'connecting' || phase === 'configuring'
  const color = isThreat ? '255,0,85' : analyzing ? '176,38,255' : '0,240,255'
  const label = isActive ? 'End Call Shield' : 'Start Call Shield'
  const tooltip = pending ? 'Securing channel…' : isActive ? 'Tap to stop streaming' : 'Tap to start protection'
  const Icon = pending ? Loader2 : isActive ? Square : Mic

  return (
    <div className="relative flex size-64 items-center justify-center sm:size-72">
      {[0, 1, 2].map((ring) => (
        <motion.span
          key={ring}
          aria-hidden="true"
          className="absolute inset-0 rounded-full border"
          style={{ borderColor: `rgba(${color},0.35)` }}
          animate={
            isActive && !reduceMotion
              ? { scale: [0.62, 1.05], opacity: [0.7, 0] }
              : { scale: 0.62 + ring * 0.12, opacity: 0.25 - ring * 0.07 }
          }
          transition={
            isActive && !reduceMotion
              ? { duration: isThreat ? 1.1 : 2.4, repeat: Infinity, delay: ring * (isThreat ? 0.36 : 0.8), ease: 'easeOut' }
              : { duration: 0.6 }
          }
        />
      ))}

      <motion.button
        type="button"
        onClick={onToggle}
        aria-label={label}
        aria-pressed={isActive}
        disabled={pending}
        whileHover={{ scale: 1.12, y: -8 }}
        whileTap={{ scale: 0.95 }}
        transition={{ type: 'spring', stiffness: 380, damping: 16 }}
        className={cn(
          'group relative z-10 flex size-40 items-center justify-center rounded-full border-2 backdrop-blur-xl outline-none transition-[box-shadow,border-color,background-color] duration-300 focus-visible:ring-4 focus-visible:ring-neon-cyan/60 disabled:cursor-wait sm:size-44',
        )}
        style={{
          borderColor: `rgba(${color},0.85)`,
          background: `radial-gradient(circle at 35% 30%, rgba(${color},0.35), rgba(9,13,22,0.85) 70%)`,
          boxShadow: `0 0 40px rgba(${color},0.45), 0 0 100px rgba(${color},0.25), inset 0 0 40px rgba(${color},0.25)`,
        }}
      >
        <span
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 rounded-full opacity-0 transition-opacity duration-300 group-hover:opacity-100"
          style={{ boxShadow: `0 0 70px rgba(${color},0.9), 0 0 160px rgba(${color},0.5), inset 0 0 60px rgba(${color},0.45)` }}
        />
        <Icon
          aria-hidden="true"
          className={cn(
            'relative size-16 text-white transition-all duration-300 group-hover:scale-110 sm:size-20',
            pending && 'animate-spin',
            isActive && !pending && 'size-12 fill-current sm:size-14',
          )}
          style={{ filter: `drop-shadow(0 0 12px rgba(${color},1)) drop-shadow(0 0 28px rgba(${color},0.8))` }}
        />
        <span
          aria-hidden="true"
          className="pointer-events-none absolute -top-12 left-1/2 -translate-x-1/2 translate-y-2 scale-90 whitespace-nowrap rounded-lg border px-3 py-1.5 font-mono text-[11px] uppercase tracking-widest text-white opacity-0 backdrop-blur-md transition-all duration-200 ease-out group-hover:translate-y-0 group-hover:scale-100 group-hover:opacity-100 group-focus-visible:translate-y-0 group-focus-visible:scale-100 group-focus-visible:opacity-100"
          style={{
            borderColor: `rgba(${color},0.6)`,
            background: 'rgba(9,13,22,0.9)',
            boxShadow: `0 0 18px rgba(${color},0.6)`,
          }}
        >
          {tooltip}
        </span>
      </motion.button>
    </div>
  )
}
