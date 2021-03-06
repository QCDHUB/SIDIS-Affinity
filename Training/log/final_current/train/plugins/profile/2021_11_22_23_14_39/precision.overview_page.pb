?	C??up?;@C??up?;@!C??up?;@	>d?r8@>d?r8@!>d?r8@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLC??up?;@?s?f?"7@1A???FX??A??fF???I?{,G(@YF$
-?>??rEagerKernelExecute 0*	?????j@2U
Iterator::Model::ParallelMapV2??O?m??!???HBB@)??O?m??1???HBB@:Preprocessing2F
Iterator::Model4M?~2ƿ?!????b?L@)?r߉Y??1eU?365@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlices??o???!1??> ?4@)s??o???11??> ?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????ؙ?!???o?z'@)?|A	??1?s???%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate~T?~O??!Tv??߷9@)L?{)<h??1???X?Z@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip!??*?C??!mRAl?"E@)?b?J!??1?'L?aN@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???5>???!{ͣ`??:@)@j'?;d?1fr???a??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?IbI??\?!"?.UT??)?IbI??\?1"?.UT??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 83.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9>d?r8@IC	_?CW@Q?????c
@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?s?f?"7@?s?f?"7@!?s?f?"7@      ??!       "	A???FX??A???FX??!A???FX??*      ??!       2	??fF?????fF???!??fF???:	?{,G(@?{,G(@!?{,G(@B      ??!       J	F$
-?>??F$
-?>??!F$
-?>??R      ??!       Z	F$
-?>??F$
-?>??!F$
-?>??b      ??!       JGPUY>d?r8@b qC	_?CW@y?????c
@?"D
&gradient_tape/model_2/dense_6/ReluGradReluGrad4?g?dֿ?!4?g?dֿ?"B
$gradient_tape/model_2/dense_7/MatMulMatMul???^"???!O?oCf??0"4
model_2/dense_7/MatMulMatMul???]a??!????x???0"B
&gradient_tape/model_2/dense_7/MatMul_1MatMulG.Ph?η?!M??????"4
model_2/dense_6/BiasAddBiasAddl?e=???!T???H"??".
model_2/dense_6/ReluRelux??U?9??!????{???"B
$gradient_tape/model_2/dense_6/MatMulMatMul??}.????!????f???0"R
1gradient_tape/model_2/dense_6/BiasAdd/BiasAddGradBiasAddGrad???^"???!{,?$?C??"4
model_2/dense_6/MatMulMatMulx??U?9??!??ީb???0"-
IteratorGetNext/_1_Send??????!??+???Q      Y@Y?18??5@a??18?S@q??j???6@y??}.????"?
both?Your program is POTENTIALLY input-bound because 83.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?22.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 