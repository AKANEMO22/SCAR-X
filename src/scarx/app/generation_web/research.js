(() => {
  const el=id=>document.getElementById(id);
  const statuses={idle:'Chưa chạy',running:'Đang chạy',completed:'Đã có kết quả',failed:'Lỗi',cancelled:'Đã dừng',stopped_without_result:'Đã dừng / thiếu kết quả'};
  let loading=false,traceLoading=false;
  let demoSignature='',historySignature='',answerSignature='',learningSignature='',jobSignature='',activitySignature='',archiveSignature='';
  function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
  async function call(path,body){
    const options={};
    if(body!==undefined){
      const status=await (await fetch('/api/status')).json();
      Object.assign(options,{method:'POST',headers:{'Content-Type':'application/json','X-SCARX-Token':status.token},body:JSON.stringify(body)});
    }
    const response=await fetch(path,options),result=await response.json();
    if(!response.ok)throw new Error(typeof result.error==='string'?result.error:result.error?.message||'Không đọc được trạng thái');
    return result;
  }
  function neurons(canvas,snapshot){
    const c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;c.clearRect(0,0,w,h);
    if(!snapshot||!Array.isArray(snapshot.spikes)){
      c.fillStyle='#92a5ab';c.font='13px Segoe UI';c.fillText('Chưa có trace từ lượt này.',20,35);return;
    }
    const count=snapshot.allocated||snapshot.spikes.length,active=snapshot.computed??count;
    const cols=Math.ceil(Math.sqrt(count*2.6)),rows=Math.ceil(count/cols);
    for(let i=0;i<count;i++){
      const x=20+(i%cols)*(w-40)/Math.max(1,cols-1),y=24+Math.floor(i/cols)*(h-46)/Math.max(1,rows-1);
      const dormant=i>=active,spike=!dormant&&snapshot.spikes[i];
      c.beginPath();c.arc(x,y,spike?6:4,0,Math.PI*2);c.strokeStyle=dormant?'#60727a':spike?'#b8efc7':'#607e8b';
      if(!dormant){c.fillStyle=spike?'#b8efc7':'#294553';c.fill();}c.stroke();
      c.fillStyle='#92a5ab';c.font='8px Consolas';c.textAlign='center';c.fillText(String(i),x,y+15);
    }
  }
  function drawDemo(demo){
    const signature=JSON.stringify([demo.status,demo.started,demo.steps?.length,demo.error]);
    if(signature===demoSignature)return;demoSignature=signature;
    el('demo-status').textContent=statuses[demo.status]||demo.status;
    el('demo-note').textContent=(demo.restored_history?'Kết quả lượt gần nhất đã lưu, không phải job còn chạy. ':'')+(demo.note||'');el('demo-error').hidden=!demo.error;if(demo.error)el('demo-error').textContent=demo.error;
    const steps=demo.steps||[],body=el('demo-steps');body.replaceChildren();
    if(!steps.length){const tr=node('tr'),td=node('td','Chưa có task được đo.','muted');td.colSpan=6;tr.append(td);body.append(tr);}
    steps.forEach((row,index)=>{
      const tr=node('tr');tr.dataset.selected=String(index===steps.length-1);
      tr.append(node('td',row.title));
      [row.words,row.active,row.allocated,row.dormant,row.grew_by?'+'+row.grew_by:'0'].forEach(v=>tr.append(node('td',String(v))));body.append(tr);
    });
    const row=steps[steps.length-1];
    if(row){
      const frames=row.frames||[],frame=frames[frames.length-1];neurons(el('demo-neurons'),frame);
      el('demo-counts').textContent=`${row.active} đang tính / ${row.allocated} đã cấp / ${row.max_neurons} tối đa · ${row.dormant} ngủ · ${row.neural_ticks} tick thực`;
      const detail=el('demo-detail');detail.replaceChildren(node('strong',row.title),node('p',row.prompt));
      detail.append(node('p',`Cùng object core: ${row.same_core_identity?'có':'chưa xác nhận'} · Cấp thêm ${row.grew_by} · Mảng neural ${(row.neural_arrays_bytes/1000).toFixed(1)} KB · ${row.timing_includes_persistence?'Cả lượt (gồm lưu trạng thái)':'Xử lý neural'} ${(row.inference_seconds*1000).toFixed(1)} ms.`));
      detail.append(node('p','Đầu ra neural chưa xác minh: '+(row.generated||'[rỗng]')),
        node('p','Mức huy động theo tải; không chứng minh task khó hơn hoặc thêm neuron làm đúng hơn. Nhịp chờ hiển thị tách khỏi thời gian tính.'));
    }else neurons(el('demo-neurons'),null);
  }
  function drawJobs(jobs){
    if(!jobs.length){el('research-current').textContent='Chưa có job mới đang được theo dõi. Các kết quả đã chạy ở bên dưới.';neurons(el('research-neurons'),null);return;}
    const job=jobs.find(j=>j.status==='running')||jobs[0],signature=JSON.stringify(job);
    if(signature===jobSignature)return;jobSignature=signature;
    const current=el('research-current'),wasOpen=!!current.querySelector('details')?.open;current.replaceChildren(node('strong',statuses[job.status]||job.status),node('p',job.title||job.run_id),node('p',job.stage));
    if(job.task)current.append(node('p','Bài: '+job.task));
    if(job.model_scope)current.append(node('p',job.model_scope));
    if(job.config?.control_only||job.result?.control_only||job.result?.word_trials?.[0]?.control_only)current.append(node('p','ĐỐI CHỨNG CÔ LẬP · BPTT chỉ nghiên cứu, không phải cơ chế học hoặc câu trả lời của lõi SCAR-X trong Hermes.','lab-verdict'));
    if(job.config?.proposal30_approved||job.result?.proposal30_approved)current.append(node('p','PROPOSAL30 ĐÃ DUYỆT NGHIÊN CỨU: một readout-errorstep + eligibility tính xuôi tại neuron. Candidate không network/time backward; bptt_control là đối chứng cô lập. Dữ liệu và phạm vi từngthínghiệm ghi bên dưới; chưa normalchat/vision/lifelong hoặc modelHermes. Nếu có bàiOrder/delayedrecency, gần100% cũng không đủ P1.','muted'));
    if(job.kind==='data_generation')current.append(node('p','Lượt tạo dữ liệu này không huấn luyện neural core. Trạng thái sử dụng về sau được ghi riêng trong báo cáo học.','muted'));
    if(job.result?.decision)current.append(node('p','Kết luận: '+job.result.decision,'lab-verdict'));
    if(job.result?.deployed===false)current.append(node('p','Chưa thay model đang dùng.','muted'));
    if(job.result?.server_stopped===true)current.append(node('p',`Server đối chứng đã tắt · ${job.result.completed??'—'}/${job.result.expected??'—'} câu · ${job.result.truncated??'—'} câu bị giới hạn độ dài.`, 'muted'));
    if(job.result?.selection?.screen_pass===false)current.append(node('p','Chưa đạt ngưỡng representation. Không thay model đang dùng.'));
    if(job.result?.primary_representation_gate===true)current.append(node('p','Qua gate representation đã đăng ký. Chỉ mở nghiên cứu P2; chưa xác nhận retention hoặc năng lực hội thoại/ảnh.','lab-verdict'));
    if(job.result?.primary_statistics){
      const wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Bài','Gain hidden · CI95%','Accuracy','Seed / Holm p','Gate'].forEach(t=>head.append(node('th',t)));table.append(head);
      Object.entries(job.result.primary_statistics).forEach(([task,s])=>{const row=node('tr');
        [task,`${(s.gain*100).toFixed(2)}đ [${(s.lower95*100).toFixed(2)}; ${(s.upper95*100).toFixed(2)}]`,(s.direct*100).toFixed(2)+'%',`${s.seeds??'—'} / ${s.holm_p===null?'chưa ghi':s.holm_p.toFixed(4)}`,s.pass?'Qua':'Chưa qua'].forEach(t=>row.append(node('td',t)));table.append(row);});
      wrap.append(table);current.append(node('p','Xác nhận độc lập · gain từ refit probe so initial/frozen · CI bootstrap theo seed.','muted'),wrap);
    }
    if(job.result?.retention_statistics){
      const source=job.result.retention_statistics,wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Cấu hình P2','Học mới','Cuối','Quên TB / tệ nhất','Probe offline','Gate'].forEach(t=>head.append(node('th',t)));table.append(head);
      const pct=v=>typeof v==='number'?(v*100).toFixed(2)+'%':'—';
      const points=v=>typeof v==='number'?(v*100).toFixed(2)+'đ':'—';
      Object.entries(source.rows).forEach(([name,r])=>{const row=node('tr');
        [name,pct(r.acquisition),pct(r.final),r.offline?'không áp dụng':`${points(r.mean_forgetting)} / ${points(r.worst_forgetting)}`,pct(r.joint_refit_probe),r.offline?'Offline control':r.gate?'Qua':'Chưa qua'].forEach(t=>row.append(node('td',t)));table.append(row);});
      wrap.append(table);current.append(node('p','P2 · trung bình3seed, chưa confirmation; offline control có future labels. Quên tính theo điểm phần trăm, không phải tỷ lệ tương đối.','muted'),wrap);
      const link=node('a','Báo cáo P2 nguồn ↗');link.href='/api/research/report?run_id='+encodeURIComponent(source.source_run);link.target='_blank';link.rel='noopener';current.append(link);
    }
    const cueSummary=job.result?.summary;
    if(cueSummary?.public_release?.canonicalreset&&cueSummary?.current_private?.canonicalreset){
      const wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Checkpoint','Trạng thái trước câu hỏi','Câu quen','48 cách nói mới','Khác reset'].forEach(t=>head.append(node('th',t)));table.append(head);
      Object.entries(cueSummary).forEach(([scope,states])=>Object.entries(states).forEach(([state,r])=>{const row=node('tr');
        [scope,state,...['known_control_exact','antigravity_exact','disagreement_vs_reset'].map(k=>typeof r[k]==='number'?(r[k]*100).toFixed(2)+'%':'chưa ghi')].forEach(t=>row.append(node('td',t)));table.append(row);}));
      wrap.append(table);current.append(node('p','Exact-match tới một đáp án kiểu mẫu, không là điểm hiểu ngôn ngữ; hai checkpoint khác weights/vocabulary/capacity.','muted'),wrap);
    }
    if(job.result?.product_trials){
      const wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Mode / thứ tự','Dấu câu','Câu quen, reset','Cách nói mới, reset','Guard mất','Screen'].forEach(t=>head.append(node('th',t)));table.append(head);
      job.result.product_trials.forEach(r=>{const row=node('tr');
        [(r.mode||r.condition)+' / '+r.curriculum_order_seed,r.surface_normalization===undefined?'theo cấu hình':r.surface_normalization?'chuẩn hóa':'gốc',
          typeof r.known_reset==='number'?(r.known_reset*100).toFixed(2)+'%':'chưa ghi',typeof r.antigravity_reset==='number'?(r.antigravity_reset*100).toFixed(2)+'%':'chưa ghi',String(r.lost_guards_total??'chưa ghi'),r.screen_pass?'Qua':'Giữ model cũ'].forEach(t=>row.append(node('td',t)));table.append(row);});
      wrap.append(table);current.append(node('p','Product clone acceptance · exact-style trên8nhóm câu; guard cũ và runtime-normalized được ghi riêng trong báo cáo nguồn. Không chứng minh normalchat hoặc lifelongP2.','muted'),wrap);
    }
    if(job.result?.word_trials){
      const source=job.result.word_trials[0];
      if(source?.source_quality_decision)current.append(node('p',(source.control_only?'Kết quả đối chứng, không là chất lượng main: ':'Quyết định chất lượng nguồn: ')+source.source_quality_decision,'lab-verdict'));
      const wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Input / hidden learning','Train16','Dev · câu tự sinh','Prior48','Dev · teacher-token'].forEach(t=>head.append(node('th',t)));table.append(head);
      const pct=v=>typeof v==='number'?(v*100).toFixed(2)+'%':'chưa ghi';
      job.result.word_trials.forEach(r=>{const row=node('tr');[r.input_encoding+' / '+(r.method||r.condition),pct(r.train16),pct(r.development),pct(r.prior),pct(r.teacher_token_accuracy)].forEach(t=>row.append(node('td',t)));table.append(row);});
      wrap.append(table);current.append(node('p',source?.control_only?'Isolated surrogate-BPTT vs frozenbaseline từv2 · cùng3pairedseeds,8styles. Có thêm graphforward/backwardcompute; không phải mainlocallearning hoặc normalchat/vision/P2retention.':'Actual word-score training · trung bình3freshmodelseeds,8stylefamilies; dev/prior đã từng xem. Teacher-token không thay điểm câu tự sinh. Audit kỹ thuật qua không thay quyết định chất lượng. Không là normalchat/vision hoặc P2retention.','muted'),wrap);
      if(source?.source_run){const link=node('a','Báo cáo chất lượng nguồn ↗');link.href='/api/research/report?run_id='+encodeURIComponent(source.source_run);link.target='_blank';link.rel='noopener';current.append(link);}
    }
    if(job.result?.credit_signal_rows){
      const wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Checkpoint nguồn','Tín hiệu đo','Cosine tái lập','Cross-signal / noise','Năng lượng noise'].forEach(t=>head.append(node('th',t)));table.append(head);
      const number=v=>typeof v==='number'?v.toPrecision(4):'thiếu / không xác định';
      job.result.credit_signal_rows.forEach(r=>{const row=node('tr');[r.source_state,r.method,number(r.replicate_cosine),number(r.cross_signal_to_pair_noise),number(r.pair_noise_energy)].forEach(t=>row.append(node('td',t)));table.append(row);});
      wrap.append(table);current.append(node('p','Audit chỉ đọc: median16input/seed, rồi trung bình3pairedseeds. Hai nhóm64noise-pairs; cosine/cross-signal là proxy tái lập, có thể âm. Baselinecrossfit lấy từ nửa đối diện nên hai mean phụ thuộc; không coi cosinecrossfit là lặp độc lập. Surrogate là biasedcontrol, không groundtruthgradient. Không có learningupdate hoặc điểm quality mới.','muted'),wrap);
      const contrasts=job.result.credit_noise_contrasts;
      if(contrasts){const box=node('div',undefined,'table-scroll'),t=node('table',undefined,'lab-table'),h=node('tr');['Nguồn','Noise crossfit / EMA · CI95%','Cosine surrogate − score · CI95%'].forEach(x=>h.append(node('th',x)));t.append(h);
        const range=v=>v?`${number(v.mean)} [${number(v.lower95)}; ${number(v.upper95)}]`:'chưa ghi';
        Object.entries(contrasts).forEach(([state,v])=>{const row=node('tr');[state,range(v.crossfit_vs_existing_noise_ratio),range(v.surrogate_minus_existing_replicate_cosine)].forEach(x=>row.append(node('td',x)));t.append(row);});box.append(t);current.append(node('p','CI bootstrap theo3modelseeds, thăm dò; case/từ không được tính thành seed. Baselinecrossfit chỉ variancecontrol, không rulemới hoặc mở lại hướng đãSTOP.','muted'),box);}
    }
    if(job.result?.association_statistics){
      const source=job.result.association_statistics,wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Local association / ablation','Train16','Dev tự sinh','Prior48','Prompt-only probe'].forEach(x=>head.append(node('th',x)));table.append(head);
      const pct=v=>typeof v==='number'?(v*100).toFixed(2)+'%':'chưa ghi';
      source.rows.forEach(r=>{const row=node('tr');[r.condition,pct(r.train16),pct(r.development),pct(r.prior),pct(r.prompt_probe)].forEach(x=>row.append(node('td',x)));table.append(row);});
      wrap.append(table);current.append(node('p','Quyết định chất lượng nguồn: '+source.decision,'lab-verdict'),node('p','Quan sát đáp án chỉ trong training, không goldinput ở inference/probe. Cùng deterministicALIF48, rawreadout và compute giữa4conditions; chỉcontextsynapses học. Mean3freshseeds/8styles; chưa là normalchat/vision/P2.','muted'),wrap);
      const link=node('a','Báo cáo association nguồn ↗');link.href='/api/research/report?run_id='+encodeURIComponent(source.source_run);link.target='_blank';link.rel='noopener';current.append(link);
    }
    if(job.result?.vision_path_rows){
      const wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Ảnh','Câu hỏi','Đáp án renderer khác nhau','Input neural giống nhau?','Xác suất giống nhau?','RGB+question prototype khác'].forEach(x=>head.append(node('th',x)));table.append(head);
      const yes=v=>v===true?'Có':v===false?'Không':'chưa đo';
      job.result.vision_path_rows.forEach(r=>{const row=node('tr');[r.scene,r.cases,r.distinct_renderer_targets,yes(r.all_questions_same_neural_input),yes(r.all_questions_same_neural_probabilities),r.prototype_distinct_multimodal_sequences].forEach(x=>row.append(node('td',String(x??'chưa đo'))));table.append(row);});
      wrap.append(table);current.append(node('p','240lần gọi Generation.vision thật trên publicclone, cùngstate/48neurons; không học. Rawdigitreadout, không VQA. Đổi câu hỏi nhưng input giống nhau là giới hạn đường vào; khác biệt ở RGBprototype không chứng minh learnedgrounding. Qwen4Btext-only không là visionreference.','muted'),wrap);
    }
    if(job.result?.deepseek_review_rows){
      const rows=job.result.deepseek_review_rows,wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),h=node('tr');
      ['Nguồn / bài / nhánh','Đúng trước','Mất sau học trên bản sao','Chỉ head gây mất','Chỉ hidden gây mất','Chỉ bias gây mất','Bước mất đầu'].forEach(x=>h.append(node('th',x)));table.append(h);
      rows.forEach(r=>{const tr=node('tr');[r.source+' / '+r.lesson+' / '+r.arm,r.baseline_pass_count,r.newly_lost_count,r.head_loss_count,r.hidden_loss_count,r.bias_loss_count,r.first_loss_step??'không mất'].forEach(x=>tr.append(node('td',String(x))));table.append(tr);});wrap.append(table);
      current.append(node('p','REVIEW DEEPSEEK · CAN THIỆP TRÊN BẢN SAO, KHÔNG DẠY PHIÊN THẬT','lab-verdict'),node('p','Mỗi nhóm được ghép đúng trọng số sau/trước học, cùng topology/vocabulary và trạng thái đo canonical. Chỉ đếm mất trên các câu đã đúng trước đó; câu vốn sai được tách riêng. Các checkpoint là trạng thái khởi đầu, không là seed độc lập. Norm/LR hoặc một cặp mất không chứng minh hidden vô ích. Đây là chẩn đoán kỹ thuật, không là model mới hoặc gate P1/P2.','muted'),wrap);
      if(job.result.metric_comparison_rows){const legacy=job.result.metric_comparison_rows.filter(r=>r.legacy_v1_correct).length,strict=job.result.metric_comparison_rows.filter(r=>r.canonical_v2_correct).length;
        current.append(node('p',`Cùng ${job.result.metric_comparison_rows.length} đầu ra neural được chấm bằng v1 (${legacy} pass) và v2 canonical (${strict} pass). Khác biệt metric không phải model học tốt hơn; v2 chỉ chấm kiểu đáp án cố định, không đo chất lượng hội thoại đa năng.`,'muted'));}
    }
    if(job.result?.word_objective_rows){
      const rows=job.result.word_objective_rows,wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Nguồn / seed / split','FREE đúng token + EOS','Từ đầu (BOS)','Từ sau với tiền tố đúng','Từ sau khi đổi câu hỏi','Gap trên cùng vị trí','Vị trí chưa đến do dừng sớm','Cosine tín hiệu đầu/sau'].forEach(x=>head.append(node('th',x)));table.append(head);
      const pct=v=>typeof v==='number'?(v*100).toFixed(2)+'%':'khôngápdụng/chưađo',points=v=>typeof v==='number'?(v*100).toFixed(2)+'đ':'chưađo';
      rows.forEach(r=>{const tr=node('tr');[r.method+'/'+r.seed+'/'+r.split,pct(r.free_exact),pct(r.tf_first),pct(r.tf_later),pct(r.wrong_later),points(r.matched_later_exposure_gap),String(r.free_gold_later_positions_censored??'chưađo'),typeof r.potential_first_later_cosine==='number'?r.potential_first_later_cosine.toFixed(3):'chỉđocorrect'].forEach(x=>tr.append(node('td',x)));table.append(tr);});wrap.append(table);
      current.append(node('p','AUDIT 38 · CHỈ ĐỌC, 0 CẬP NHẬT NEURAL','lab-verdict'),node('p','FREE dùng từ mà model tự sinh trước đó. GOLDPREFIX cung cấp tiền tố đúng để chẩn đoán, KHÔNG là câu model tự trả lời. FREE ở đây đòi đúng mọi token và EOS, chặt hơn điểm normalized style của vòng 36 bỏ dấu cuối câu; model không bị đổi điểm. Đổi/xóa câu hỏi có thể tạo đầu vào ngoài phân phối. Gap chỉ tính trên cùng vị trí FREE đã đến; vị trí bị dừng sớm được ghi riêng. Hướng cập nhật tiềm năng ở checkpoint cuối không phải ΔW quá khứ hoặc tín hiệu đã áp dụng. Bảng vòng 36 phía dưới là số đo nguồn cũ, không là kết quả huấn luyện mới.','muted'),wrap);
      const structure=job.result.corpus_structure;
      if(structure)current.append(node('p',`Corpus hữu hạn: ${structure.targets} mẫu đáp án, ${structure.unique_first_words} từ đầu khác nhau; ${structure.later_positions}/${structure.total_positions_uniform_templates} vị trí nằm sau bước đầu. ${pct(structure.nonempty_prefix_determines_next_fraction)} tiền tố chuẩn không rỗng xác định từ kế tiếp. Đây là thuộc tính corpus, không phải model đã hiểu hoặc bảng tra được triển khai.`,'muted'));
    }
    if(job.result?.credit_alignment_statistics){
      const source=job.result.credit_alignment_statistics,wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Seed / split','NLL trước','ΔNLL hướng local','ΔNLL đảo dấu','ΔNLL control hoán vị','Tỉ lệ đúng hướng','Độ lớn control/Thật (hoán vị)'].forEach(x=>head.append(node('th',x)));table.append(head);
      const num=v=>typeof v==='number'?v.toFixed(5):'chưa đo',pct=v=>typeof v==='number'?(v*100).toFixed(2)+'%':'chưa đo';
      source.rows.forEach(r=>{const row=node('tr');[r.seed+' / '+r.split,num(r.nll_before),num(r.delta_direction),num(r.delta_signflip),num(r.delta_permuted_mean),pct(r.alignment_fraction),typeof r.norm_match_ratio_permuted==='number'?r.norm_match_ratio_permuted.toFixed(2)+'×':'chưa đo'].forEach(x=>row.append(node('td',x)));table.append(row);});
      wrap.append(table);
      const m=source.dev_macro||{},ci=source.intervals||{};
      const range=v=>v?`${num(v.mean)} [${num(v.lower95)}; ${num(v.upper95)}]`:'chưa đo';
      current.append(node('p','AUDIT 39 · CHỈ ĐỌC · 0 CẬP NHẬT NEURAL','lab-verdict'),
        node('p',`Áp chính hướng local (trung bình endpoint, Adam/clip của checkpoint) lên BẢN SAO dùng một lần rồi đo lại NLL teacher-forced thật. ΔNLL âm = loss GIẢM. Dev macro: đúng hướng ${pct(m.alignment_fraction)}, ΔNLL ${num(m.delta_direction)}. Chênh so đảo dấu: ${range(ci.gain_vs_signflip)}.`,`muted`),
        node('p','CẢNH BÁO PHƯƠNG PHÁP: Adam chuẩn hóa theo từng phần tử, nên control Gaussian khớp norm TRƯỚC Adam bị áp lực lớn hơn thật rất nhiều, và control hoán vị cũng lớn hơn. Vì vậy chỉ số "gain so với control" bị nhiễu bởi độ lớn; chỉ control ĐẢO DẤU khớp chính xác độ lớn áp dụng. Đây là lý do gain so đảo dấu mới là bằng chứng chính, không phải gain so random.','muted'),
        node('p',`Held-out (hướng gộp từ TRAIN, áp một lần, chấm trên dev): ΔNLL trung bình ${num(source.heldout?.mean_dev_delta)}. Đây là con số duy nhất nói về tổng quát hóa, và nó rất nhỏ so với mức cải thiện trên train.`,'muted'),
        node('p','ĐỌC BẢNG: tỉ lệ đúng hướng ở TRAIN thấp (9–11%) KHÔNG phải quy tắc sai. NLL train đã bão hòa ≈0,0105–0,0157 nên không còn dư địa giảm loss; dev còn dư địa (0,76–0,94) và là nơi tín hiệu thể hiện. Chỉ đọc phần dev khi đánh giá hướng tín hiệu.','muted'),
        node('p','Không huấn luyện, không tối ưu, không head refit, không Torch. Checkpoint nguồn và trọng số chính không đổi. Bảng này KHÔNG phải câu trả lời model và không phải bằng chứng năng lực ngôn ngữ.','muted'),wrap);
      if(source.flags)current.append(node('pre',JSON.stringify(source.flags,null,2)),node('p','Flag là mô tả chẩn đoán đã đăng ký trước, không phải cổng chất lượng của dự án.','muted'));
      const link=node('a','Báo cáo đầy đủ audit 39 ↗');link.href='/api/research/report?run_id='+encodeURIComponent(source.run_id||job.run_id);link.target='_blank';link.rel='noopener';current.append(link);
    }
    if(job.result?.visual_information_rows){
      const rows=job.result.visual_information_rows,wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),h=node('tr');
      ['Nguồn / seed','Từ đầu neural','PROBE từ đầu','PROBE thuộc tính','PROBE nhãn tráo','Clip ±5','Rank48'].forEach(x=>h.append(node('th',x)));table.append(h);
      const pct=v=>typeof v==='number'?(v*100).toFixed(2)+'%':'chưa đo';
      rows.forEach(r=>{const row=node('tr');[r.family+' / '+r.seed,...['actual_BOS_firstword','auxiliary_firstword_probe','auxiliary_semantic_probe','auxiliary_shuffled_semantic_probe','head_clip_fraction'].map(k=>pct(r[k])),String(r.actual_head_feature_rank??'chưa đo')].forEach(x=>row.append(node('td',x)));table.append(row);});
      wrap.append(table);current.append(node('p','READ ONLY · FIXED AUXILIARY PROBE · KHÔNG HUẤN LUYỆN SYNAPSE','lab-verdict'),node('p','Đo tại bước sinh từ đầu tiên từ checkpoint32/33. PROBE là classifier chẩn đoán ngoài core, không phải câu trả lời model hoặc năng lực đã triển khai. Dùng đúng1536lượt nhãn train đã thấy; semanticprobe là phân loại, không sinh câu. Archive câu đầy đủ vẫn giữ đầu ra nguồn33 với thời điểm cũ, không giả là inference mới của audit.','muted'),wrap);
      if(job.result.diagnostic_flags)current.append(node('pre',JSON.stringify(job.result.diagnostic_flags,null,2)),node('p','Flag componentcollapse bao gồm nhánh chỉcâuhỏi vốn không chứa RGB; không tự là phát hiện mất thông tin của toàn core. Phải đọc vị trí và counterfactual bên dưới.','muted'));
      const detail=node('details'),stageWrap=node('div',undefined,'table-scroll'),stageTable=node('table',undefined,'lab-table'),sh=node('tr');
      detail.append(node('summary','Exactcollisions và counterfactual theo đường thông tin'));
      ['Nguồn / vị trí','Trần thuộc tính trên dev đã đo','Cặp khác ảnh bị đồng nhất','Cặp khác câu hỏi bị đồng nhất'].forEach(x=>sh.append(node('th',x)));stageTable.append(sh);
      (job.result.information_collapse_rows||[]).forEach(r=>{const row=node('tr');[r.family+' / '+r.stage+(r.expected_missingmodality_branch?' (chỉ câu hỏi, không có RGB)':''),pct(r.dev_semantic_ceiling_mean),pct(r.image_pair_collapse_mean),pct(r.question_pair_collapse_mean)].forEach(x=>row.append(node('td',x)));stageTable.append(row);});
      stageWrap.append(stageTable);detail.append(node('p','Trần oracle chỉ theo các vector trùng byte trên tập hữu hạn hiện tại; không là dung lượng trí nhớ hay trần năng lực chung. Mất thông tin ở một nhánh không chứng minh toàn trạng thái đã mất thông tin.','muted'),stageWrap);current.append(detail);
    }
    const directWordSource=job.result?.direct_word_statistics;
    if(directWordSource||(job.config?.protocol==='experiments/direct-word-bridge-v1.md'&&job.result?.summary)){
      const table=node('table',undefined,'lab-table'),head=node('tr'),wrap=node('div',undefined,'table-scroll');
      ['Nhánh / 3seed','Câu tự sinh dev','Câu train','Resethead probe gain','Khôi phục hidden đầu'].forEach(x=>head.append(node('th',x)));table.append(head);
      const pct=v=>typeof v==='number'?(v*100).toFixed(2)+'%':'chưa đo',pts=v=>typeof v==='number'?(v*100).toFixed(2)+'đ':'chưa đo';
      Object.entries(directWordSource?.summary||job.result.summary).forEach(([method,r])=>{const row=node('tr');[method,pct(r.macro_development),pct(r.macro_train),pts(r.probe_gain),pct(r.restore_initial_hidden_dev)].forEach(x=>row.append(node('td',x)));table.append(row);});
      wrap.append(table);current.append(node('p','WORD BRIDGE · closed8stylepilot, khôngnormalchat/P1/P2 hoặc mainHermes. Câu trả lời tự sinh từng từ, targets chỉtraining/chấm. BPTT làcontrolcôlập; resetheadprobe làclassifierdiagnostic, khôngdecoder.','muted'),wrap);
      if(directWordSource){const link=node('a','Kết quả neuralword nguồn: '+directWordSource.decision+' ↗');link.href='/api/research/report?run_id='+encodeURIComponent(directWordSource.source_run);link.target='_blank';link.rel='noopener';current.append(link);}
    }
    if(job.result?.visual_word_statistics){
      const source=job.result.visual_word_statistics,wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Inputcondition','Macro dev','Tổ hợp mới','Màuđơn','Hìnhđơn','Đếm','Màubêntrái',...(source.direct_eligibility?['Macro train','Probe gain']:[])].forEach(x=>head.append(node('th',x)));table.append(head);
      const pct=v=>typeof v==='number'?(v*100).toFixed(2)+'%':'khôngápdụng/chưađo';
      source.rows.forEach(r=>{const row=node('tr');[r.condition,...['macro_exact','novel_composition_macro','single_color','single_shape','object_count','left_color'].map(k=>pct(r[k])),...(source.direct_eligibility?[pct(r.macro_train),typeof r.probe_gain==='number'?(r.probe_gain*100).toFixed(2)+'đ':'chưađo']:[])].forEach(x=>row.append(node('td',x)));table.append(row);});
      wrap.append(table);current.append(node('p','Quyết định chất lượng nguồn: '+source.decision+(source.interpretation?' · '+source.interpretation:''),'lab-verdict'),node('p',source.direct_eligibility?'DIRECT VISUAL37 · cùngALIF48/forward/cache32, approvedreadoutfeedback/localcacheeligibility35. 3freshseeds trên864devgeometriccases đãtừngxem. no_image/no_question làtrainedinputablations, zero/frozen chỉheadhọc; BPTT làcontrolcôlập. Probe làfirstwordclassifierdiagnostic, không câu trả lời. Khôngmain/normalvision/P1/P2 hoặc scale.':source.control_only?'ĐỐI CHỨNG ALIF-BPTT CÔ LẬP · 3pairedseeds, hidden học bằng surrogateBPTT; head output-local nhưbaseline. Baselinefrozen dùng kết quả32 đã kiểmparity; cùngexperiences nhưng graphcompute thêm. Không main/locallearning/normalvision/P2 hoặc proposal30.':'3freshmodelseeds, macro theo kind/language; 864dev geometriccases, câu hỏi/tổ hợp táchtrain. Wordoutputsneural thực, hiddenFROZEN, không hostformat từclasslabel. Không normalvision/P1hiddenlearning/P2retention hoặc mainimport.','muted'),wrap);
      const box=node('div',undefined,'table-scroll'),t=node('table',undefined,'lab-table'),h=node('tr');['Pairedcontrast','Gain/drop · CI95%'].forEach(x=>h.append(node('th',x)));t.append(h);
      const points=v=>typeof v==='number'?(v*100).toFixed(2)+'đ':'chưađo';
      Object.entries(source.contrasts).forEach(([group,values])=>Object.entries(values).forEach(([key,v])=>{const row=node('tr');row.append(node('td',group+' / '+key),node('td',`${points(v.mean)} [${points(v.lower95)}; ${points(v.upper95)}]`));t.append(row);}));box.append(t);current.append(box);
      const link=node('a','Báo cáo jointvisualword nguồn ↗');link.href='/api/research/report?run_id='+encodeURIComponent(source.source_run);link.target='_blank';link.rel='noopener';current.append(link);
    }
    if(cueSummary?.offline_joint_full?.baseline?.probe!==undefined){
      const wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Nguồn','Dynamics view','Head cũ','Refit probe','Gain'].forEach(t=>head.append(node('th',t)));table.append(head);
      Object.entries(cueSummary).forEach(([scope,views])=>Object.entries(views).forEach(([view,r])=>{const row=node('tr');
        [scope,view,...['direct','probe','gain'].map(k=>typeof r[k]==='number'?(r[k]*100).toFixed(2)+(k==='gain'?'đ':'%'):'chưa ghi')].forEach(t=>row.append(node('td',t)));table.append(row);}));
      wrap.append(table);current.append(node('p','Dynamics inference-view diagnostic; weights bất biến, refit head chưa triển khai.','muted'),wrap);
    }
    if(cueSummary&&Object.values(cueSummary).some(v=>v&&v['8']&&v['64']&&typeof v['8']['12']==='number')){
      const wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),head=node('tr');
      ['Feature nguồn','MC rollouts','Tick2','Tick6','Tick12'].forEach(t=>head.append(node('th',t)));table.append(head);
      Object.entries(cueSummary).forEach(([name,s])=>{if(!s?.['8']||!s?.['64'])return;
        ['8','64'].forEach(mc=>{const row=node('tr');[name,mc,...['2','6','12'].map(t=>Number.isFinite(s[mc][t])?(s[mc][t]*100).toFixed(2)+'%':'chưa ghi')].forEach(t=>row.append(node('td',t)));table.append(row);});});
      wrap.append(table);current.append(node('p','Chẩn đoán thông tin · refit head, không train core; MC64 dùng thêm8× lượt inference, không phải cải thiện cùng compute hoặc model đã triển khai.','muted'),wrap);
    }
    if(job.result?.control_summary){
      const wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'lab-table'),thead=node('thead'),labels=node('tr');
      ['Bài đo','Đối chứng','Accuracy','Hidden probe gain'].forEach(v=>labels.append(node('th',v)));thead.append(labels);table.append(thead);
      const body=node('tbody');
      Object.entries(job.result.control_summary).forEach(([task,conditions])=>Object.entries(conditions).forEach(([condition,value])=>{
        const row=node('tr'),accuracy=typeof value==='number'?value:value?.direct,gain=value?.probe_gain;
        [task,condition,typeof accuracy==='number'?(accuracy*100).toFixed(2)+'%':'chưa ghi',typeof gain==='number'?(gain*100).toFixed(2)+' điểm %':'chưa ghi'].forEach(v=>row.append(node('td',v)));body.append(row);
      }));table.append(body);wrap.append(table);current.append(node('p','So sánh các cấu hình của lượt thử · accuracy trên bộ bài đã công bố.','muted'),wrap);
    }
    const resources=job.process_resources||{},progress=job.progress||{};
    el('research-progress').max=progress.total||1;el('research-progress').value=progress.completed||0;
    el('research-metrics').textContent=[progress.total?`${progress.completed||0}/${progress.total} lượt`:null,
      resources.resident_tree_bytes?`${(resources.resident_tree_bytes/1e6).toFixed(1)} MB RSS`:null,
      job.process_alive?'Đã xác nhận tiến trình còn sống':'Không báo chạy chỉ từ file trạng thái'].filter(Boolean).join(' · ');
    const advanced=node('details'),summary=node('summary','Cấu hình và số đo'),pre=node('pre',JSON.stringify({configuration:job.config,metrics:job.metrics,result:job.result},null,2));advanced.open=wasOpen;advanced.append(summary,pre);current.append(advanced);
    const events=el('research-events');events.replaceChildren();
    (job.events||[]).slice(-8).reverse().forEach(event=>events.append(node('p',new Date(event.time*1000).toLocaleTimeString('vi-VN')+' · '+event.stage)));
    neurons(el('research-neurons'),job.neural);
    if(job.neural){const n=job.neural;el('research-neural-counts').textContent=`${n.computed??n.spikes?.length} đang tính / ${n.allocated??n.spikes?.length} đã cấp · tick ${n.tick??'—'} · mẫu thật gần nhất, không phải độc thoại nội tâm.`;}
    else el('research-neural-counts').textContent='Lượt này chưa công bố mẫu neuron; không tạo trace giả.';
  }
  function drawHistory(history){
    const signature=JSON.stringify(history);if(signature===historySignature)return;historySignature=signature;
    const box=el('research-history'),opened=new Set([...box.querySelectorAll('details[open]')].map(d=>d.dataset.run));box.replaceChildren();
    history.forEach(item=>{
      const details=node('details'),summary=node('summary',item.title+' · '+(item.result.decision||'Đã có kết quả'));
      details.dataset.run=item.run_id;details.open=opened.has(item.run_id);
      const link=node('a','Mở báo cáo nguồn ↗');link.href=item.report_url;link.target='_blank';link.rel='noopener';
      details.append(summary,link,node('pre',JSON.stringify(item.result,null,2)));box.append(details);
    });
  }
  async function poll(){
    if(loading)return;loading=true;
    try{
      const data=await call('/api/research');el('research-connection').textContent='● Trạng thái được cập nhật';el('research-error').hidden=true;
      const live=data.live_capacity;el('capacity-live').textContent=`Lõi neural đang mở: ${live.active} đang tính / ${live.allocated} đã cấp / ${live.maximum} tối đa.`;
      if(live.workload)el('capacity-live').textContent+=` Lượt gần nhất: ${live.workload.input_characters} ký tự, ${live.workload.active_neurons} neuron.`;
      drawDemo(data.capacity_demo);drawJobs(data.jobs);drawHistory(data.history);drawArchive(data.answer_archive);
    }catch(error){el('research-connection').textContent='○ Chưa kết nối';el('research-error').hidden=false;el('research-error').textContent=error.message;}
    finally{loading=false;}
  }
  const stamp=row=>new Date(row.time*1000).toLocaleString('vi-VN');
  function field(parent,title,text){const box=node('div',undefined,'trace-field');box.append(node('small',title),node('p',text??'Chưa có'));parent.append(box);}
  function measuredImages(parent,row){
    if(!row.image_data_uri&&!row.encoded_image_data_uri)return;
    const images=node('div',undefined,'answer-images');
    [['Ảnh đầu vào thực',row.image_data_uri],['64 pixel xám thực đi vào neuron',row.encoded_image_data_uri]].forEach(([title,uri])=>{
      if(typeof uri!=='string'||uri.length>15000||!/^data:image\/png;base64,[A-Za-z0-9+/]+=*$/.test(uri))return;
      const frame=node('figure'),picture=node('img');picture.src=uri;picture.alt=title;picture.width=128;picture.height=128;picture.loading='lazy';frame.append(picture,node('figcaption',title));images.append(frame);
    });parent.append(images);if(row.image_source)parent.append(node('p',row.image_source,'muted'));
  }
  function drawArchive(archive){
    const signature=archive?.signature||'';if(signature===archiveSignature)return;archiveSignature=signature;
    const box=el('archive-cases');box.replaceChildren();
    if(!archive?.rows?.length){el('archive-count').textContent='Chưa có bản đo đã lưu';el('archive-note').textContent='Chưa có tệp đầu ra đối chứng hoàn tất cho lượt đang theo dõi. Trang không chạy model để lấp dữ liệu.';return;}
    el('archive-count').textContent=`${archive.case_count} câu · seed${archive.seed}`;
    const directVisual=archive.learning_mechanism==='direct_visual_bridge',directWord=archive.learning_mechanism==='direct_word_bridge',eligibility=archive.learning_mechanism==='direct_readout_eligibility',observed=archive.learning_mechanism==='observed_association',vision=archive.learning_mechanism==='vision_path_audit',visualControl=archive.learning_mechanism==='isolated_visual_word_bptt',visualWords=archive.learning_mechanism==='visual_word_readout',visualInputs=visualWords||visualControl||directVisual,bptt=archive.learning_mechanism==='isolated_bptt'||visualControl||((eligibility||directWord||directVisual)&&archive.method==='bptt_control');
    el('archive-count').textContent=eligibility?`${archive.case_count} tensorcases · seed${archive.seed} · ${archive.task}/${archive.method}`:vision?`${archive.case_count} image-question calls`:`${archive.case_count} câu · seed${archive.seed}`;
    el('archive-note').textContent=`${visualControl?'ĐỐI CHỨNG ALIF-BPTT ẢNH–CÂU HỎI · hidden học surrogateBPTT, head local; KHÔNG MAIN/locallearning/normalvision':visualWords?'JOINT VISUAL WORDS · '+archive.condition+' · hiddenđóngbăng, chỉreadouthọc; no_image/no_question làablationinput, khôngnormalvision':vision?'VISION PATH AUDIT · publicrelease clone, chỉdigitclassification, không huấnluyện hoặc VQA':observed?'OBSERVED-ASSOCIATION · '+archive.condition+' · training có quan sát đáp án, inference không nhận gold, không backward':bptt?'ĐỐI CHỨNG BPTT CÔ LẬP':'BẢN ĐO THỰC · '+(archive.scope||'chưa công bố cơ chế')}, tách khỏi phiên trong Hermes. Toàn bộ case công khai đã đo trong tệp gần nhất, gồm câu sai; không chọn theo điểm. Hồ sơ lưu ${new Date(archive.record_saved_at_utc).toLocaleString('vi-VN')} (thời điểm lưu tệp, không phải thời gian từng inference).`;
    archive.rows.forEach((row,index)=>{
      const card=node('details',undefined,'trace-card');card.open=index===0;
      card.append(node('summary',`${row.case_id} · ${vision?'raw digitreadout, không chấm style':eligibility?(row.style_match===true?'đúng nhãn số':'sai nhãn số'):row.style_match===true?'khớp style mẫu':row.style_match===false?'khác style mẫu':'chưa ghi điểm style'}`));
      field(card,'Đầu vào thực sự gửi bản thử nghiệm',row.prompt);
      measuredImages(card,visualInputs?{...row,...archive.images?.[row.scene_id]}:row);
      if(visualInputs)card.append(node('p',row.novel_composition?'Tổ hợp cảnh chưa train; cách hỏi dev riêng.':'Tổ hợp đãtrain, geometry/cách hỏi dev riêng.','muted'));
      const grid=node('div',undefined,'answer-columns');field(grid,directVisual?`ALIF visualwords thực · ${archive.method} · KHÔNG MAIN/NORMALVISION`:eligibility?`ALIF binaryhead thực · ${archive.method} · KHÔNG CHAT/MAIN`:visualControl?'Đối chứng ALIF-BPTT sinh từ thực · KHÔNG MAIN':visualWords?'Từ neuralreadout thực · không class-nameformatter':vision?'Neural digitreadout thực · không là câu trả lời ngôn ngữ':observed?'ALIF observed-association sinh thực tế':bptt?'Đối chứng ALIF-BPTT sinh thực tế':'Đầu ra neural đã ghi nhận',row.generated||'[Đầu ra rỗng]');
      field(grid,eligibility?'Nhãn từ generator công khai':vision?'Thuộc tính đúng theo manifest renderer':'Một đáp án style mẫu · không là chân lý tự động',row.target);field(grid,'Qwen · nguồn đo nếu có',row.qwen?.generated??row.qwen_missing_reason??'Chưa có bản Qwen đo cùng prompt');card.append(grid);
      if(row.qwen)card.append(node('p',`${row.qwen.scope} · ${row.qwen.measured_at_utc||row.qwen.run} · kết thúc: ${row.qwen.finish_reason}${row.qwen.finish_reason==='length'?' (bị cắt)':''}`,'muted'));
      box.append(card);
    });
    if(eligibility)el('archive-note').textContent=`PROPOSAL30 RESEARCH · ${archive.phase}/${archive.task}/${archive.method}. Actual float32[12,8]→ALIF→binaryhead, targetkhông vào inference. ${bptt?'BPTT chỉđốichứng côlập.':'Một readout-errorstep với localforwardeligibility; zero/frozen chỉheadhọc.'} Đây là phân loại số, KHÔNG normalchat/vision hoặc modelHermes. Tệp gần nhất theo thời điểm ${new Date(archive.record_saved_at_utc).toLocaleString('vi-VN')}, không chọn theo điểm; không giảQwencomparison.`;
    if(directWord)el('archive-note').textContent=`PROPOSAL30 WORD BRIDGE · ${archive.method} · 64DEVclosedstyleprompts. Neuralwordhead tựsinh chuỗi từ, không lấy target/intent/language làm input. ${bptt?'BPTT chỉđốichứng côlập.':'Ứngviên directfeedback/localeligibility; zero/frozen chỉheadhọc.'} Chưa là normalchat/vision/P1/P2 hoặc mainHermes. Tệp gần nhất theo thời điểm ${new Date(archive.record_saved_at_utc).toLocaleString('vi-VN')}, kểcảcâusai. Qwen nếu có làbản đo cùngprompt với thờiđiểm riêng, không nhãn học.`;
    if(directVisual)el('archive-note').textContent=`DIRECT VISUAL37 · ${archive.method} · same32ALIF/input/cache, approved30localcacheeligibility. ${bptt?'BPTT chỉcontrolcôlập.':'zero/frozen chỉheadhọc; correct/shuffled/no_image/no_question cóhiddenlearning theo method.'} Đây là864DEVgeometriccases đãđo, khôngnormalphotos/main/P1/P2. Ảnhhiểnthị làảnhnguồn; methodno_image/no_question đãzero modality trong actualneuralinput. Tệp gần nhất ${new Date(archive.record_saved_at_utc).toLocaleString('vi-VN')}, không chọn theo điểm. Không cóQwenvision giả.`;
    const link=node('a','Báo cáo của lượt nguồn ↗');link.href='/api/research/report?run_id='+encodeURIComponent(archive.run_id);link.target='_blank';link.rel='noopener';box.append(link);
  }
  function drawActivities(rows){
    const signature=JSON.stringify(rows);if(signature===activitySignature)return;activitySignature=signature;
    const box=el('activity-trace');box.replaceChildren();
    if(!rows.length){box.append(node('p','Chưa có công việc kỹ thuật trong phần nhật ký đang giữ.','muted'));return;}
    const labels={started:'Đã bắt đầu',completed:'Đã hoàn tất',failed:'Chưa đạt / lỗi'};
    rows.slice(0,12).forEach((row,index)=>{
      const card=node('details',undefined,'trace-card');card.open=index===0;
      card.append(node('summary',`${stamp(row)} · ${labels[row.status]||'Đã ghi nhận'} · ${row.title}`));
      field(card,'Công việc kỹ thuật',row.detail);
      if(row.evidence){const proof=node('details');proof.append(node('summary','Bằng chứng đã ghi'),node('pre',JSON.stringify(row.evidence,null,2)));card.append(proof);}
      const link=node('a','Lượt công việc / báo cáo nguồn ↗');link.href='/api/research/report?run_id='+encodeURIComponent(row.run_id);link.target='_blank';link.rel='noopener';card.append(link);box.append(card);
    });
  }
  function drawAnswers(rows){
    const signature=JSON.stringify(rows);if(signature===answerSignature)return;answerSignature=signature;
    const box=el('answer-trace');if(!rows.length)return;box.replaceChildren();
    rows.forEach((row,index)=>{
      const card=node('details',undefined,'trace-card');card.open=index===0;
      card.append(node('summary',`${stamp(row)} · ${row.scope||'Bản thử nghiệm'} · ${row.case_id||row.prompt||row.run_id}`));
      field(card,'Đầu vào / ngữ cảnh',row.messages?row.messages.map(m=>`${m.role}: ${m.text}`).join('\n'):row.prompt);
      field(card,'Đầu vào thực sự gửi SCAR-X',row.prompt);
      field(card,'Phạm vi đầu vào',row.input_scope);
      measuredImages(card,row);
      const grid=node('div',undefined,'answer-columns');
      field(grid,row.output_type==='digit_classifier'?'Neural digitreadout thực · không phải ngôn ngữ':row.control_only||/ISOLATED|BPTT/i.test(row.scope||'')?'Đối chứng cô lập sinh thực tế · không phải main':/PROPOSAL30/.test(row.scope||'')?'Prototype ALIF phân loại số thực · KHÔNG CHAT/MAIN':'SCAR-X sinh thực tế',row.generated===''?'[Đầu ra rỗng]':row.generated);
      field(grid,'Một đáp án mẫu · không phải nhãn tự học',row.target??'Không có đáp án mẫu cho lượt này');
      field(grid,'Qwen3-4B · bản đo đã lưu',row.qwen?.generated??row.qwen_missing_reason??'Chưa chạy Qwen cho cùng bài này');card.append(grid);
      if(row.qwen)card.append(node('p',`${row.qwen.scope} · ${row.qwen.measured_at_utc||row.qwen.run} · kết thúc: ${row.qwen.finish_reason}${row.qwen.finish_reason==='length'?' (đã bị cắt)':''}`,'muted'));
      card.append(node('p',`${row.neural_ticks??'—'} tick · ${row.active??'—'}/${row.allocated??'—'} neuron · ${row.seconds===undefined?'chưa ghi':row.seconds===0?'dưới độ phân giải đồng hồ cũ':(row.seconds*1000).toFixed(2)+' ms'} · checkpoint ${row.checkpoint??'chưa ghi'}`,'muted'));box.append(card);
    });
  }
  const fmt=v=>typeof v==='number'?v.toPrecision(4):'chưa ghi';
  const weights=value=>value?Object.entries(value).map(([k,v])=>`${k}: ${fmt(v)}`).join(' · '):'chưa ghi';
  function drawLearning(rows,controller){
    el('learning-counters').textContent=`${controller.accepted} giữ · ${controller.rejected} hoàn tác · ${controller.queued} bài chờ · ${controller.updates} bước cập nhật token`;
    const signature=JSON.stringify(rows);if(signature===learningSignature)return;learningSignature=signature;
    if(!rows.length)return;const box=el('learning-trace');box.replaceChildren();
    const points=rows.filter(r=>typeof r.loss_mean==='number').slice(0,40).reverse();
    const canvas=el('learning-loss'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
    if(points.length){
      const max=Math.max(.01,...points.map(r=>r.loss_mean));ctx.strokeStyle='#60727a';ctx.beginPath();ctx.moveTo(45,12);ctx.lineTo(45,135);ctx.lineTo(975,135);ctx.stroke();
      ctx.fillStyle='#92a5ab';ctx.font='12px Segoe UI';ctx.fillText(max.toFixed(2),3,20);ctx.fillText('0',20,139);
      points.forEach((r,i)=>{const x=52+(points.length===1?0:i/(points.length-1))*910,y=132-r.loss_mean/max*112;
        ctx.fillStyle=r.rolled_back?'#f6a477':'#b8efc7';ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();});
      el('loss-caption').textContent=`${points.length} giao dịch có loss gần nhất, từ cũ → mới; có thể gồm nhiều bài khác nhau. Chấm cam = thử rồi rollback. Không nối thành loss huấn luyện của một dataset.`;
    }else el('loss-caption').textContent='Log cũ chưa ghi loss; không dựng đường giả.';
    rows.slice(0,30).forEach((row,index)=>{
      const card=node('details',undefined,'trace-card');card.open=index===0;
      const accepted=row.accepted??row.action!=='online-reject';
      const copyOnly=String(row.scope||'').startsWith('COPY ONLY');
      card.append(node('summary',`${stamp(row)} · ${copyOnly?'CHẨN ĐOÁN TRÊN BẢN SAO':accepted?(row.guard_checked?'GUARD GIỮ':'ĐÃ GIỮ · guard chưa ghi/không dùng'):'ROLLBACK'} · loss ${fmt(row.loss_mean)} · ${row.prompt||row.run_id}`));
      field(card,'Nguồn / bài học',`${row.scope||'Bản thử nghiệm'} · ${row.prompt||'chưa ghi'} → ${row.target||'chưa ghi'}`);
      field(card,'Lượt học / loss',`${row.passes??row.repeats??'chưa ghi'} lượt · loss TB ${fmt(row.loss_mean)} · ${(row.losses||[]).map(fmt).join(' → ')||'chưa ghi từng lượt'}`);
      field(card,'Cập nhật synapse',`${row.update_steps??'chưa ghi'} bước ${row.update_unit||'token'} · số trọng số đổi ròng: ${row.changed_synapses?Object.entries(row.changed_synapses).map(([k,v])=>`${k}: ${v}`).join(' · '):'chưa ghi'}${row.hidden_update_steps!==undefined?' · '+row.hidden_update_steps+' hidden updates':''}`);
      field(card,'ΔW thử trước guard',weights(row.attempted_weight_delta));field(card,copyOnly?'ΔW trên bản sao, không lưu vào phiên thật':'ΔW giữ lại sau giao dịch',weights(row.weight_delta));
      field(card,'Guard',row.reason||`${row.guard_probe_count??'chưa ghi'} probe · mất ${row.guard_lost??'chưa ghi'} probe · ${row.rolled_back?'đã hoàn tác':accepted?'giao dịch được giữ':'chưa rõ'}`);
      if(row.stopped_identical_proposal)card.append(node('p',`Đã dừng sau ${row.identical_guard_rejections} đề xuất giống hệt bị guard từ chối. Trọng số giữ nguyên; không hạ ngưỡng để ép học.`, 'lab-verdict'));
      if(row.before!==undefined)field(card,'Sinh trước học → sau lần thử',`${row.before||'[rỗng]'} → ${row.after_attempt||'[rỗng]'}${row.rolled_back?' (bản thử này đã bị hoàn tác)':''}`);
      box.append(card);
    });
  }
  async function pollTrace(){
    if(traceLoading)return;traceLoading=true;
    try{const trace=await call('/api/trace');drawActivities(trace.activities||[]);drawAnswers(trace.answers);drawLearning(trace.learning,trace.controller);
      const connection='● Tự cập nhật · '+new Date().toLocaleTimeString('vi-VN');el('trace-connection').textContent=connection;el('activity-connection').textContent=connection;el('trace-error').hidden=true;}
    catch(error){el('trace-connection').textContent='○ Mất kết nối';el('activity-connection').textContent='○ Mất kết nối';el('trace-error').hidden=false;el('trace-error').textContent=error.message;}
    finally{traceLoading=false;}
  }
  poll();pollTrace();setInterval(poll,1500);setInterval(pollTrace,1500);
})();
