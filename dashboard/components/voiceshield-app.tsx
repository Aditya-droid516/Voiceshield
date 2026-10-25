'use client'

import { AnimatePresence, motion } from 'framer-motion'
import { useState } from 'react'
import { Toaster } from 'sonner'
import { RealtimeCallShield } from '@/components/call-shield/realtime-call-shield'
import { FileForensicsStudio } from '@/components/forensics/file-forensics-studio'
import { type AppTab, SiteHeader } from '@/components/site-header'

export function VoiceShieldApp() {
  const [tab, setTab] = useState<AppTab>('forensics')

  return (
    <>
      <SiteHeader tab={tab} onTabChange={setTab} />
      <main className="mx-auto w-full max-w-6xl">
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            id={`panel-${tab}`}
            role="tabpanel"
            aria-labelledby={`tab-${tab}`}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
          >
            {tab === 'forensics' ? <FileForensicsStudio /> : <RealtimeCallShield />}
          </motion.div>
        </AnimatePresence>
      </main>
      <Toaster
        position="bottom-right"
        theme="dark"
        toastOptions={{
          classNames: {
            toast:
              '!rounded-xl !border !border-neon-cyan/40 !bg-slate-900/70 !text-white !backdrop-blur-[20px] !shadow-[0_0_30px_-6px_rgba(0,240,255,0.6)]',
            error: '!border-neon-crimson/60 !shadow-[0_0_30px_-6px_rgba(255,0,85,0.8)] [&_[data-icon]]:!text-neon-crimson',
            success: '!border-neon-emerald/60 !shadow-[0_0_30px_-6px_rgba(0,255,102,0.7)] [&_[data-icon]]:!text-neon-emerald',
            description: '!text-slate-300',
          },
        }}
      />
    </>
  )
}
