'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Real microphone capture hook.
 *
 * Uses getUserMedia + MediaRecorder to record from the mic and produce an
 * audio Blob (webm/opus or mp4 depending on browser support) plus a File
 * ready to POST to the speech pipeline.
 *
 * Why a hook: the voice page needs to coordinate permission state, recording
 * state, elapsed time, and MediaRecorder lifecycle across renders, and all of
 * that has to clean up correctly on unmount or the browser keeps the mic
 * indicator lit. Centralizing it here keeps the page component readable and
 * makes the cleanup invariants testable in isolation.
 */

export type RecorderStatus =
  | 'idle'
  | 'requesting_permission'
  | 'recording'
  | 'stopped'
  | 'error'

export interface VoiceRecorderError {
  code:
    | 'not_supported'
    | 'permission_denied'
    | 'no_microphone'
    | 'recorder_failed'
    | 'no_data'
  message: string
}

export interface UseVoiceRecorderResult {
  status: RecorderStatus
  elapsedSeconds: number
  error: VoiceRecorderError | null
  audioBlob: Blob | null
  audioUrl: string | null
  /** Best-effort media type for the recording (e.g. audio/webm;codecs=opus). */
  mimeType: string | null
  isSupported: boolean
  start: () => Promise<void>
  stop: () => void
  reset: () => void
}

/** Pick the first MIME type the browser can actually record. */
function pickMimeType(): string | null {
  if (typeof MediaRecorder === 'undefined') return null
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4', // Safari
  ]
  for (const type of candidates) {
    try {
      if (MediaRecorder.isTypeSupported(type)) return type
    } catch {
      // isTypeSupported can throw on some browsers; try the next candidate.
    }
  }
  return ''
}

export function useVoiceRecorder(): UseVoiceRecorderResult {
  const [status, setStatus] = useState<RecorderStatus>('idle')
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [error, setError] = useState<VoiceRecorderError | null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [mimeType, setMimeType] = useState<string | null>(null)

  const mediaStreamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const stoppedManuallyRef = useRef(false)

  const isSupported =
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== 'undefined'

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const releaseStream = useCallback(() => {
    if (mediaStreamRef.current) {
      for (const track of mediaStreamRef.current.getTracks()) track.stop()
      mediaStreamRef.current = null
    }
  }, [])

  const reset = useCallback(() => {
    clearTimer()
    releaseStream()
    recorderRef.current = null
    chunksRef.current = []
    stoppedManuallyRef.current = false
    setStatus('idle')
    setElapsedSeconds(0)
    setError(null)
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioBlob(null)
    setAudioUrl(null)
  }, [audioUrl, clearTimer, releaseStream])

  const start = useCallback(async () => {
    if (!isSupported) {
      setError({
        code: 'not_supported',
        message:
          'Audio recording is not supported in this browser. Try the latest Chrome, Firefox, or Edge over HTTPS.',
      })
      setStatus('error')
      return
    }

    // Reset any previous recording state before starting a new one.
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioBlob(null)
    setAudioUrl(null)
    setError(null)
    setElapsedSeconds(0)
    chunksRef.current = []
    stoppedManuallyRef.current = false

    setStatus('requesting_permission')

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
    } catch (err) {
      const name = (err as DOMException)?.name
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        setError({
          code: 'permission_denied',
          message: 'Microphone permission was denied. Enable mic access in your browser to record.',
        })
      } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
        setError({
          code: 'no_microphone',
          message: 'No microphone was found. Connect a mic and try again.',
        })
      } else {
        setError({
          code: 'permission_denied',
          message: `Could not access the microphone: ${(err as Error).message || name || 'unknown error'}.`,
        })
      }
      setStatus('error')
      return
    }

    const type = pickMimeType()
    let recorder: MediaRecorder
    try {
      recorder = type ? new MediaRecorder(stream, { mimeType: type }) : new MediaRecorder(stream)
    } catch (err) {
      releaseStream()
      setError({
        code: 'recorder_failed',
        message: `Could not start the recorder: ${(err as Error).message}.`,
      })
      setStatus('error')
      return
    }

    mediaStreamRef.current = stream
    recorderRef.current = recorder
    setMimeType(recorder.mimeType || type || null)

    recorder.ondataavailable = (event: BlobEvent) => {
      if (event.data && event.data.size > 0) chunksRef.current.push(event.data)
    }

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, {
        type: recorder.mimeType || type || 'audio/webm',
      })
      if (blob.size === 0) {
        setError({
          code: 'no_data',
          message: 'Recording produced no audio data. Check your microphone and try again.',
        })
        setStatus('error')
        releaseStream()
        return
      }
      const url = URL.createObjectURL(blob)
      setAudioBlob(blob)
      setAudioUrl(url)
      setStatus('stopped')
      releaseStream()
    }

    recorder.onerror = () => {
      setError({
        code: 'recorder_failed',
        message: 'The recorder reported an error. Please try again.',
      })
      setStatus('error')
      releaseStream()
    }

    // Collect data every 250ms so onstop has chunks even on very short clips.
    recorder.start(250)
    setStatus('recording')

    clearTimer()
    timerRef.current = setInterval(() => {
      setElapsedSeconds((s) => s + 1)
    }, 1000)
  }, [audioUrl, clearTimer, isSupported, releaseStream])

  const stop = useCallback(() => {
    stoppedManuallyRef.current = true
    clearTimer()
    const recorder = recorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      // onstop will assemble the blob and flip status to 'stopped'.
      recorder.stop()
    } else {
      releaseStream()
      setStatus('stopped')
    }
  }, [clearTimer, releaseStream])

  // Cleanup on unmount: stop recorder, release mic, revoke object URL.
  useEffect(() => {
    return () => {
      clearTimer()
      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        try {
          recorderRef.current.stop()
        } catch {
          /* ignore — best-effort cleanup */
        }
      }
      if (mediaStreamRef.current) {
        for (const track of mediaStreamRef.current.getTracks()) track.stop()
      }
    }
  }, [clearTimer])

  return {
    status,
    elapsedSeconds,
    error,
    audioBlob,
    audioUrl,
    mimeType,
    isSupported,
    start,
    stop,
    reset,
  }
}

/**
 * Convert a recorded Blob into a File with a sensible name and extension
 * derived from the MIME type, ready to POST as multipart/form-data.
 */
export function blobToAudioFile(blob: Blob, recordingId: string): File {
  const ext = extensionForMime(blob.type)
  const name = `prism-voice-${recordingId}.${ext}`
  return new File([blob], name, { type: blob.type || 'audio/webm' })
}

function extensionForMime(mime: string): string {
  const lower = (mime || '').toLowerCase()
  if (lower.includes('webm')) return 'webm'
  if (lower.includes('ogg')) return 'ogg'
  if (lower.includes('mp4') || lower.includes('m4a')) return 'm4a'
  if (lower.includes('wav')) return 'wav'
  return 'webm'
}
