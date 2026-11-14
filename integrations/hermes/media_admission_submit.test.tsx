import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { useSubmitPrompt } from './submit'
import type { GatewayRequest } from './utils'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  retainGatewayForSessionTurn: vi.fn(async () => () => undefined)
}))

beforeEach(() => {
  window.localStorage.clear(); $sessions.set([]); $sessionStates.set({})
  $connection.set(null); $activeGatewayProfile.set('default')
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

function setup(fresh = false) {
  let state = createClientSessionState()
  const active = { current: fresh ? null : 'runtime-a' as string | null }
  const stored = { current: fresh ? null : 'stored-a' as string | null }
  const request = vi.fn(async (_method: string, params?: Record<string, unknown>) =>
    ({ admission_id: params?.submission_id, submission_id: params?.submission_id, status: 'started' }) as never)
  const deps = {
    activeSessionIdRef: active, selectedStoredSessionIdRef: stored, busyRef: { current: false }, copy: en.desktop,
    createBackendSessionForSend: vi.fn(async () => { active.current = 'created-runtime'; stored.current = 'created-runtime'; return 'created-runtime' }),
    getRoutedStoredSessionId: () => null, getRuntimeIdForStoredSession: () => active.current,
    getRouteToken: () => 'same-route', requestGateway: request as GatewayRequest,
    runtimeIdByStoredSessionIdRef: { current: new Map(fresh ? [] : [['stored-a', 'runtime-a']]) },
    resumeStoredSession: vi.fn(),
    syncAttachmentsForSubmit: vi.fn(async (sessionId: string) => ({ sessionId, attachments: [] })),
    updateSessionState: vi.fn((_sid: string, updater: (s: ReturnType<typeof createClientSessionState>) => ReturnType<typeof createClientSessionState>) => { state = updater(state); return state })
  }
  return { deps, request, stored }
}

it('waits for request metadata after fresh session creation and sends the same immutable input id', async () => {
  const { deps, request } = setup(true)
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  const prepare = vi.fn(async () => { await gate })
  const hook = renderHook(() => useSubmitPrompt(deps as never))
  let sent!: Promise<boolean>
  await act(async () => { sent = hook.result.current('same question', { submission_id: 'draft-A', onSessionPrepared: prepare }) })
  await waitFor(() => expect(prepare).toHaveBeenCalledTimes(1))
  expect(deps.createBackendSessionForSend).toHaveBeenCalledTimes(1)
  expect(prepare).toHaveBeenCalledWith({ runtimeSessionId: 'created-runtime', storedSessionId: 'created-runtime', submissionId: 'draft-A', text: 'same question' })
  expect(request.mock.calls.filter(([method]) => method === 'prompt.submit')).toHaveLength(0)
  await act(async () => { release(); expect(await sent).toBe(true) })
  const params = request.mock.calls.find(([method]) => method === 'prompt.submit')![1]!
  expect(params.session_id).toBe('created-runtime'); expect(params.submission_id).toBe('draft-A')
  expect(params.text).toBe('same question'); expect(JSON.stringify(params)).not.toContain('onSessionPrepared')
})

it('a preparation failure admits no input and an explicit retry keeps its submission identity', async () => {
  const { deps, request } = setup()
  const prepare = vi.fn().mockRejectedValueOnce(new Error('owned metadata failure')).mockResolvedValue(undefined)
  const hook = renderHook(() => useSubmitPrompt(deps as never))
  await act(async () => { expect(await hook.result.current('question', { submission_id: 'stable-input', onSessionPrepared: prepare })).toBe(false) })
  expect(request.mock.calls.filter(([method]) => method === 'prompt.submit')).toHaveLength(0)
  await act(async () => { expect(await hook.result.current('question', { submission_id: 'stable-input', onSessionPrepared: prepare })).toBe(true) })
  expect(prepare.mock.calls.map(([identity]) => identity.submissionId)).toEqual(['stable-input', 'stable-input'])
  expect(request.mock.calls.find(([method]) => method === 'prompt.submit')![1]!.submission_id).toBe('stable-input')
})

it('navigation during preparation cannot submit the original draft into a newly focused chat', async () => {
  const { deps, request, stored } = setup()
  const hook = renderHook(() => useSubmitPrompt(deps as never))
  await act(async () => {
    expect(await hook.result.current('question', { submission_id: 'owned-input', onSessionPrepared: async () => { stored.current = 'another-chat' } })).toBe(false)
  })
  expect(request.mock.calls.filter(([method]) => method === 'prompt.submit')).toHaveLength(0)
})
