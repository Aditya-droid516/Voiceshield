'use client'

import { motion } from 'framer-motion'
import { FileAudio, PhoneCall, ShieldCheck } from 'lucide-react'
import { GlowIcon } from '@/components/glow-icon'
import { HealthBadge } from '@/components/health-badge'
import { cn } from '@/lib/utils'

export type AppTab = 'forensics' | 'call'

const TABS: { id: AppTab; label: string; short: string; icon: typeof FileAudio }[] = [
  { id: 'forensics', label: 'File Forensics Studio', short: 'Forensics', icon: FileAudio },
  { id: 'call', label: 'Live Call Shield', short: 'Call Shield', icon: PhoneCall },
]

export function SiteHeader({ tab, onTabChange }: { tab: AppTab; onTabChange: (tab: AppTab) => void }) {
  return (
    <header className="glass sticky top-4 z-30 mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-3 rounded-2xl px-4 py-3">
      <div className="flex items-center gap-3">
        <GlowIcon icon={ShieldCheck} label="VoiceShield" tone="cyan" />
        <div className="leading-tight">
          <p className="text-sm font-semibold tracking-wide text-white">VoiceShield</p>
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Enterprise API v2.1</p>
        </div>
      </div>

      <nav aria-label="Primary" className="order-3 w-full sm:order-none sm:w-auto">
        <div role="tablist" aria-label="Workspace" className="flex gap-1 rounded-xl border border-white/10 bg-white/5 p-1">
          {TABS.map(({ id, label, short, icon: Icon }) => {
            const selected = tab === id
            return (
              <motion.button
                key={id}
                type="button"
                role="tab"
                id={`tab-${id}`}
                aria-selected={selected}
                aria-controls={`panel-${id}`}
                onClick={() => onTabChange(id)}
                whileHover={{ scale: 1.06, y: -3 }}
                whileTap={{ scale: 0.97 }}
                transition={{ type: 'spring', stiffness: 400, damping: 18 }}
                className={cn(
                  'relative flex flex-1 items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium outline-none transition-colors duration-300 focus-visible:ring-2 focus-visible:ring-neon-cyan/70 sm:flex-none',
                  selected ? 'text-neon-cyan' : 'text-muted-foreground hover:text-white',
                )}
              >
                {selected && (
                  <motion.span
                    layoutId="tab-pill"
                    aria-hidden="true"
                    className="absolute inset-0 rounded-lg border border-neon-cyan/50 bg-neon-cyan/10 shadow-[0_0_20px_-4px_rgba(0,240,255,0.6)]"
                    transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                  />
                )}
                <Icon
                  className="relative size-4 transition-[filter] duration-300 hover:drop-shadow-[0_0_10px_rgba(0,240,255,0.95)]"
                  aria-hidden="true"
                />
                <span className="relative hidden md:inline">{label}</span>
                <span className="relative md:hidden">{short}</span>
              </motion.button>
            )
          })}
        </div>
      </nav>

      <HealthBadge />
    </header>
  )
}
