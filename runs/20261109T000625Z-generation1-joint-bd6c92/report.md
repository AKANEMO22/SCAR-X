# Generation 1 joint training

One48-ALIF core, shared input/recurrent/readout arrays. Two output slices; no model router.

| Seed | Dev text | Dev digits | Greeting acceptance | All text acceptance | Digits diagnostic |
|---|---:|---:|---:|---:|---:|
| 17 | 0.375 | 0.861 | 0.500 | 0.333 | 0.883 |
| 29 | 0.312 | 0.872 | 0.500 | 0.222 | 0.894 |

Release seed17 selected by development only.

## Release acceptance outputs

- xin chào cậu → Chào lời. (target: Chào bạn.)
- chào scar-x nhé → Chào bạn. (target: Chào bạn.)
- chào bạn ơi → Chào bạn. (target: Chào bạn.)
- hi friend → Hello tôi đang hoạt tôi. (target: Hello.)
- hello scar-x! → Hello. (target: Hello.)
- hey there → Tôi đang. (target: Hello.)
- tên của bạn là gì vậy → Tôi đang hoạt động bạn. (target: Tôi là SCAR-X.)
- cho tôi biết bạn là ai → Hello. (target: Tôi là SCAR-X.)
- cảm ơn bạn nhiều → Rất vui được giúp bạn. (target: Rất vui được giúp bạn.)
- thank you friend → Bạn. (target: Rất vui được giúp bạn.)
- hẹn gặp lại bạn nhé → Bạn. (target: Tạm biệt bạn.)
- goodbye scar-x → Tạm biệt bạn. (target: Tạm biệt bạn.)
- bạn có đang hoạt động không → Tôi đang hoạt động. (target: Tôi đang hoạt động.)
- are you there scar-x → Hello. (target: Tôi đang hoạt động.)
- hãy nói khả năng của bạn → Tôi đang học trả lời bạn. (target: Tôi đang học trả lời.)
- what can you do scar-x → Rất vui đang học trả đang (target: Tôi đang học trả lời.)
- làm sao để dạy bạn → Tôi đang hoạt động. (target: Bạn có thể dạy tôi.)
- can you learn scar-x → Dạy tôi đang hoạt. (target: Bạn có thể dạy tôi.)

Product diagnostic, tiny authored families; no representation/retention/general language proof. Previous research PIVOT_NO_SCALE remains.
