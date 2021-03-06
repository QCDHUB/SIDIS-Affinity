?	KZ???=@KZ???=@!KZ???=@	???㽰?????㽰??!???㽰??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLKZ???=@֍wGƾ8@1??d?z???As??/٨?I?=?4a@Y????(??rEagerKernelExecute 0*	v?V?h@2U
Iterator::Model::ParallelMapV2l\??ϰ?!5D\*
?@@)l\??ϰ?15D\*
?@@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?g@???!Jk???4@)?g@???1Jk???4@:Preprocessing2F
Iterator::Model???????!;?	S#J@)'?
b???1?홾?D3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?k%t?ġ?!??4?xq1@)$}ZE??1??20@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????????!iy?D}~9@)W'g(?x??1y8l?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????aN??!??V???G@)-σ??v{?1E???9?
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap4?/.Ui??!WOz??:@)T?:?g?1?^?ȩ??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor${??!Ud?!???????)${??!Ud?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 83.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???㽰??I??+?t?W@Q?w???
@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	֍wGƾ8@֍wGƾ8@!֍wGƾ8@      ??!       "	??d?z?????d?z???!??d?z???*      ??!       2	s??/٨?s??/٨?!s??/٨?:	?=?4a@?=?4a@!?=?4a@B      ??!       J	????(??????(??!????(??R      ??!       Z	????(??????(??!????(??b      ??!       JGPUY???㽰??b q??+?t?W@y?w???
@?"D
&gradient_tape/model_2/dense_6/ReluGradReluGrad[?NC%S??![?NC%S??"B
$gradient_tape/model_2/dense_7/MatMulMatMul?,:???!0??>????0"4
model_2/dense_7/MatMulMatMulYN9l?ŷ?!.m??F??0"B
&gradient_tape/model_2/dense_7/MatMul_1MatMul?y?x?6??!?;?????"4
model_2/dense_6/BiasAddBiasAdd??E?)???!??- ?v??".
model_2/dense_6/ReluRelu??'????!?;?Z??"-
IteratorGetNext/_1_Send?ʳ????!??r*??"B
$gradient_tape/model_2/dense_6/MatMulMatMulZ?a?????!Z%?I????0"R
1gradient_tape/model_2/dense_6/BiasAdd/BiasAddGradBiasAddGradZ??ZJ??!???????"4
model_2/dense_6/MatMulMatMul??
1]???!ہ?Ǽ???0Q      Y@Y?18??5@a??18?S@q?sI?9;@y'ɪ????"?
both?Your program is POTENTIALLY input-bound because 83.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?27.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 