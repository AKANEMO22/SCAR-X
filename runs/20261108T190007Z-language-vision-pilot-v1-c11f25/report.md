# Language/vision pilot v1

Language exact test: 0.000; limited candidate pass: False

| Prompt | Neural generation | Expected supervision |
|---|---|---|
| chào bạn | Bạn, | chào bạn, tôi là scar-x. |
| xin chào nhé | Bạn, tôi là scar-x | chào bạn, tôi là scar-x. |
| cho tôi biết tên của bạn | Bạn, tôi. | tôi là scar-x, một mạng neuron đang học. |
| ai đang trả lời tôi |  | tôi là scar-x, một mạng neuron đang học. |
| cám ơn bạn nhiều | Tôi là scar-x, tôi bạn | cảm ơn bạn, tôi sẽ tiếp tục học. |
| cảm ơn vì đã trả lời | Bạn, tôi là scar-x. | cảm ơn bạn, tôi sẽ tiếp tục học. |
| hẹn bạn lần sau | Tôi. | tạm biệt bạn, hẹn gặp lại. |
| bái bai | Bạn, tôi. | tạm biệt bạn, hẹn gặp lại. |
| bạn có chắc là đúng | Tôi. | tôi đang học và có thể trả lời sai. |
| có nên tin mọi câu bạn nói | Tôi là scar-x | tôi đang học và có thể trả lời sai. |
| tôi dạy cho bạn nhé | Bạn, bạn, một mạng neuron đang học. | bạn có thể dạy tôi bằng ví dụ đúng. |
| tôi có thể đưa ví dụ để bạn học | Tôi là scar-x | bạn có thể dạy tôi bằng ví dụ đúng. |
| bạn hãy nhìn hình giúp tôi | Bạn, tôi là scar-x | bạn hãy gửi ảnh để tôi thử nhận dạng. |
| phân tích bức ảnh nhé | Bạn, tôi là scar-x. | bạn hãy gửi ảnh để tôi thử nhận dạng. |
| ai thắng trận bóng hôm nay | Bạn, tôi. | tôi chưa biết, bạn hãy dạy tôi. |
| hãy giải thích điều bạn chưa học | Tôi là scar-x | tôi chưa biết, bạn hãy dạy tôi. |

Digits classification diagnostic only. Does not establish screenshot/UI understanding, OCR, visual QA or general supports_vision.
- Vision full: test accuracy 0.664
- Vision readout: test accuracy 0.664

Resources: {'peak_rss_bytes': 162795520, 'peak_commit_or_vms_bytes': 127000576, 'os_process_peak_working_set_bytes': 163905536, 'os_process_peak_pagefile_commit_bytes': 127868928, 'samples': 102, 'interval_seconds': 0.5, 'limit_bytes': 2000000000, 'observed_over_limit': False, 'elapsed_seconds': 29.86705030000303, 'hard_cap_enforced': False, 'igpu_measured': False, 'energy_joules': None, 'note': 'Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated.'}

Production was not silently replaced. This is not a general chat/vision model.
