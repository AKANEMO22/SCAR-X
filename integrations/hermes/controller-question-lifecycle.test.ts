import { afterEach, expect, it, vi } from 'vitest'
import { $clarifyRequests, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { handleInputRequestEvent } from './input-requests'
import type { GatewayEventContext } from './types'

afterEach(() => clearClarifyRequest())

it('settles an idle controller question without waiting for an LLM tool continuation', () => {
  setClarifyRequest({sessionId:'qa',requestId:'wire',question:'1+1?',choices:null})
  let state={busy:false,needsInput:true,awaitingResponse:true,streamId:'question'}
  const upsert=vi.fn()
  const ctx={event:{type:'clarify.resolved'},payload:{request_id:'wire',user_response:'2'},sessionId:'qa',occurredAt:1,
    deps:{activeSessionIdRef:{current:'qa'},sessionInterrupted:()=>false,upsertToolCall:upsert,
      updateSessionState:(_sid:string,update:(value:typeof state)=>typeof state)=>state=update(state)}} as unknown as GatewayEventContext
  expect(handleInputRequestEvent(ctx)).toBe(true)
  expect($clarifyRequests.get().qa).toBeUndefined()
  expect(state.needsInput).toBe(false)
  expect(state.streamId).toBeNull()
  expect(upsert.mock.calls[0][1].result.user_response).toBe('2')
})

it('does not close another question when an older resolved event arrives late', () => {
  setClarifyRequest({sessionId:'qa',requestId:'newer',question:'new question',choices:null})
  const update=vi.fn(),upsert=vi.fn()
  const ctx={event:{type:'clarify.resolved'},payload:{request_id:'older',user_response:'2'},sessionId:'qa',occurredAt:1,
    deps:{activeSessionIdRef:{current:'qa'},sessionInterrupted:()=>false,upsertToolCall:upsert,updateSessionState:update}} as unknown as GatewayEventContext
  handleInputRequestEvent(ctx)
  expect($clarifyRequests.get().qa?.requestId).toBe('newer')
  expect(update).not.toHaveBeenCalled()
  expect(upsert).not.toHaveBeenCalled()
})

it('settles after respond acknowledgement already cleared the pending store', () => {
  clearClarifyRequest()
  let state={busy:false,needsInput:true,awaitingResponse:true,streamId:'question'}
  const upsert=vi.fn()
  const ctx={event:{type:'clarify.resolved'},payload:{request_id:'wire',user_response:'4',question:'2+2?',standalone:true},sessionId:'qa',occurredAt:1,
    deps:{activeSessionIdRef:{current:'qa'},sessionInterrupted:()=>false,upsertToolCall:upsert,
      updateSessionState:(_sid:string,update:(value:typeof state)=>typeof state)=>state=update(state)}} as unknown as GatewayEventContext
  handleInputRequestEvent(ctx)
  expect(state.needsInput).toBe(false)
  expect(state.streamId).toBeNull()
  expect(upsert.mock.calls[0][1].result.user_response).toBe('4')
})
