'use client'

import { AnimatePresence, motion } from 'framer-motion'
import { FileAudio, UploadCloud } from 'lucide-react'
import { useId, useRef, useState } from 'react'
import { GlowIcon } from '@/components/glow-icon'
import { ACCEPTED_EXTENSIONS, formatBytes } from '@/lib/forensics'
import { cn } from '@/lib/utils'

type DropzoneProps = {
  onFile: (file: File) => void
  loading: boolean
  fileName?: string
  fileSize?: number
}

export function Dropzone({ onFile, loading, fileName, fileSize }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const hintId = useId()

  return (
    <motion.div
      whileHover={loading ? undefined : { scale: 1.01, y: -3 }}
      transition={{ type: 'spring', stiffness: 300, damping: 22 }}
      onDragOver={(e) => {
        e.preventDefault()
        if (!loading) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        const file = e.dataTransfer.files?.[0]
        if (file && !loading) onFile(file)
      }}
      className={cn(
        'glass relative flex min-h-64 flex-col items-center justify-center gap-4 overflow-hidden rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-all duration-300',
        dragging
          ? 'border-neon-cyan bg-neon-cyan/10 shadow-[0_0_60px_-10px_rgba(0,240,255,0.8)]'
          : 'border-neon-cyan/30 hover:border-neon-cyan/70 hover:shadow-[0_0_40px_-12px_rgba(0,240,255,0.7)]',
      )}
    >
      <AnimatePresence>
        {loading && (
          <motion.div
            aria-hidden="true"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="pointer-events-none absolute inset-0"
          >
            <motion.div
              className="absolute inset-x-0 h-24 bg-gradient-to-b from-transparent via-neon-cyan/25 to-transparent"
              animate={{ top: ['-25%', '100%'] }}
              transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut', repeatType: 'reverse' }}
            >
              <div className="absolute inset-x-0 top-1/2 h-0.5 bg-neon-cyan shadow-[0_0_20px_4px_rgba(0,240,255,0.9)]" />
            </motion.div>
            <div className="absolute inset-0 bg-[linear-gradient(rgba(0,240,255,0.06)_1px,transparent_1px),linear-gradient(90deg,rgba(0,240,255,0.06)_1px,transparent_1px)] bg-[size:24px_24px]" />
          </motion.div>
        )}
      </AnimatePresence>

      <GlowIcon icon={loading ? FileAudio : UploadCloud} label={loading ? 'Scanning' : 'Upload audio'} tone={loading ? 'violet' : 'cyan'} />

      <div className="relative flex flex-col gap-1">
        <p className="text-lg font-semibold text-white">
          {loading ? 'Running deep forensic scan…' : 'Drop an audio file to analyze'}
        </p>
        <p id={hintId} className="text-sm text-muted-foreground">
          {loading && fileName
            ? `${fileName} · ${formatBytes(fileSize)}`
            : `${ACCEPTED_EXTENSIONS.join(' · ')} — up to 25 MB`}
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={`${ACCEPTED_EXTENSIONS.join(',')},audio/*`}
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          e.target.value = ''
        }}
      />
      <button
        type="button"
        disabled={loading}
        aria-describedby={hintId}
        onClick={() => inputRef.current?.click()}
        className="relative rounded-xl border border-neon-cyan/60 bg-neon-cyan/10 px-5 py-2.5 font-mono text-xs font-semibold uppercase tracking-[0.2em] text-neon-cyan transition-all duration-300 ease-out hover:-translate-y-1.5 hover:scale-105 hover:border-neon-cyan hover:shadow-[0_0_28px_rgba(0,240,255,0.8)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neon-cyan/70 disabled:pointer-events-none disabled:opacity-50"
      >
        {loading ? 'Scanning…' : 'Browse files'}
      </button>
    </motion.div>
  )
}
