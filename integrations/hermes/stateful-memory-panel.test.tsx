import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { contextBarLabel, usageContextLabel } from '@/lib/statusbar'
import type { UsageStats } from '@/types/hermes'

import { ContextUsagePanel } from './context-usage-panel'

afterEach(cleanup)

it('does not display a transcript occupancy percentage as neural memory capacity', () => {
  const usage: UsageStats = {
    calls: 1, input: 0, output: 0, total: 0,
    context_mode: 'backend_state', input_limit_characters: 2048,
    context_max: 4096, context_used: 1800, context_percent: 44
  }
  expect(usageContextLabel(usage)).toBe('Chưa có lượt nạp')
  expect(contextBarLabel(usage)).toBe('')
  render(<ContextUsagePanel breakdown={null} loading={false} usage={usage} />)
  expect(screen.getByText(/Chưa có lượt nạp trong phiên này/)).toBeTruthy()
  expect(screen.getByText(/Dung lượng trí nhớ và kỹ năng: chưa đo/)).toBeTruthy()
  expect(screen.queryByText(/44%/)).toBeNull()
})

it('changes actual load and neuron counts for each request without inventing memory fullness', () => {
  const usage: UsageStats = {calls:1,input:0,output:0,total:0,context_mode:'backend_state',input_limit_characters:2048,
    request_load:{input_characters:80,input_utf8_bytes:110,image_count:1,active_neurons:64,allocated_neurons:96,max_neurons:128,
      neural_input_channels:240,neural_input_buffer_bytes:960,prompt_features_nonzero:30,neural_image_pixels:64}}
  expect(usageContextLabel(usage)).toBe('Nạp 80 ký tự + 1 ảnh · 64 neuron')
  render(<ContextUsagePanel breakdown={null} loading={false} usage={usage} />)
  expect(screen.getByText(/Neuron đang tính: 64 · đã cấp: 96 · trần: 128/)).toBeTruthy()
  expect(screen.getByText(/960 byte/)).toBeTruthy()
  expect(usageContextLabel({...usage,request_load:{input_characters:2,input_utf8_bytes:2,image_count:0,active_neurons:48}})).toBe('Nạp 2 ký tự · 48 neuron')
})
