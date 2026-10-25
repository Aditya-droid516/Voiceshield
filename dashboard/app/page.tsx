import { VoiceShieldApp } from '@/components/voiceshield-app'

export default function Page() {
  return (
    <div className="relative min-h-dvh overflow-hidden bg-gradient-to-b from-void to-void-2">
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute -left-40 -top-40 size-[36rem] rounded-full bg-neon-cyan/15 blur-[140px]" />
        <div className="absolute -right-32 top-1/3 size-[32rem] rounded-full bg-neon-violet/20 blur-[140px]" />
        <div className="absolute -bottom-48 left-1/3 size-[30rem] rounded-full bg-neon-cyan/10 blur-[160px]" />
      </div>

      <div className="relative flex flex-col gap-8 px-4 pb-16 sm:px-6">
        <VoiceShieldApp />
      </div>
    </div>
  )
}
