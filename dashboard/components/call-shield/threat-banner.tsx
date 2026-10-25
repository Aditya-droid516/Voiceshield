'use client'

import { AnimatePresence, motion } from 'framer-motion'
import { ShieldAlert } from 'lucide-react'
import { GlowIcon } from '@/components/glow-icon'

export function ThreatBanner({ show, alert, score }: { show: boolean; alert: string; score: number }) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          role="alert"
          initial={{ opacity: 0, y: -16, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -12, scale: 0.97 }}
          transition={{ type: 'spring', stiffness: 320, damping: 24 }}
          className="threat-pulse flex flex-col gap-3 rounded-2xl border-2 border-neon-crimson bg-neon-crimson/15 p-4 backdrop-blur-xl sm:flex-row sm:items-center"
        >
          <GlowIcon icon={ShieldAlert} label="Deepfake detected" tone="crimson" className="bg-neon-crimson/20" />
          <div className="flex-1">
            <p className="font-mono text-[10px] uppercase tracking-[0.3em] text-neon-crimson">High Priority · Call Shield</p>
            <p className="text-pretty text-base font-semibold text-white sm:text-lg" style={{ textShadow: '0 0 14px #FF0055' }}>
              {alert}
            </p>
            <p className="text-xs text-white/70">
              Smoothed score exceeded 0.75 for 2 consecutive hops. Treat the caller as unverified and do not share
              sensitive information.
            </p>
          </div>
          <p className="font-mono text-3xl font-bold tabular-nums text-neon-crimson" style={{ textShadow: '0 0 20px #FF0055' }}>
            {score.toFixed(2)}
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
