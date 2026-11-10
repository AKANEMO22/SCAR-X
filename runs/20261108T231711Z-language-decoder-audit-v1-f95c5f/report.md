# Language decoder audit

Greedy devexact=0.188; beam devexact=0.125; gate=False

| Prompt | Greedy | Beam |
|---|---|---|
| xin chào scar-x | Chào bạn, tôi là scar-x. | Chào bạn. |
| chào buổi chiều | Chào bạn, tôi là scar-x. | Tôi là scar-x. |
| bạn có tên là gì | Tôi là scar-x tôi là. | Tôi là scar-x, một mạng neuron đang học. |
| tự giới thiệu đi | Tôi. | Tôi. |
| cảm ơn scar-x | Bạn, bạn, bạn, bạn, bạn, bạn, bạn, bạn, bạn. | Bạn, bạn, tôi sẽ tiếp tục học. |
| tôi cảm ơn bạn | Cảm ơn bạn, bạn có thể trả lời sai. | . |
| tạm biệt nhé | Tôi là scar-x, hẹn gặp lại. | Tôi là scar-x, hẹn gặp lại. |
| tôi phải đi rồi | Tạm biệt bạn hãy dạy tôi bằng ví dụ đúng. | Tạm biệt bạn, tôi bằng ví dụ đúng. |
| câu trả lời có thể sai không | Tôi. | Tôi. |
| bạn thông minh chưa | Cảm ơn cảm ơn. | Cảm ơn cảm ơn. |
| cách dạy scar-x | Bạn có thể dạy tôi là scar-x. | Bạn có thể dạy tôi là scar-x. |
| tôi sẽ dạy bạn | Bạn có thể dạy tôi bằng ví dụ đúng. | Bạn có thể dạy tôi bằng ví dụ đúng. |
| hãy xem ảnh | Bạn, tôi. | Bạn, tôi. |
| tôi muốn bạn nhận dạng hình | Tôi bằng ví. | Tôi đang học. |
| giúp tôi làm bài khó | Tôi sẽ tiếp tục học. | Tôi sẽ tiếp tục học. |
| trả lời câu hỏi khó | Tôi bằng ví dụ đúng. | Tôi. |

No weight updates or response lookup. Test unopened, production unchanged.
