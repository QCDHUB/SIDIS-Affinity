?	?o+?f"@?o+?f"@!?o+?f"@	w?`?	U@w?`?	U@!w?`?	U@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?o+?f"@???̯&	@1o?$????A??-=????I>$|?o?@Y????T???rEagerKernelExecute 0*	?S㥛?j@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicec??????!?4???Q8@)c??????1?4???Q8@:Preprocessing2U
Iterator::Model::ParallelMapV2??'G???!̓??/~6@)??'G???1̓??/~6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???????!?3?I7@)W{?l??1?»s!c5@:Preprocessing2F
Iterator::Model?{???!ģ+?&?B@)b/????1xg?l:8.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate-??2:??!)??;?u?@)??OI??1??΄B?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??-Y??!;\?|?2O@)1#?=??1Ȉn??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????J??!??{?H?@@)0H????p?1ي9?2	??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorϣ????p?!x@D?+l??)ϣ????p?1x@D?+l??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?48.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t34.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9v?`?	U@I???B??T@Qn(!? %@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???̯&	@???̯&	@!???̯&	@      ??!       "	o?$????o?$????!o?$????*      ??!       2	??-=??????-=????!??-=????:	>$|?o?@>$|?o?@!>$|?o?@B      ??!       J	????T???????T???!????T???R      ??!       Z	????T???????T???!????T???b      ??!       JGPUYv?`?	U@b q???B??T@yn(!? %@?"@
"gradient_tape/model/dense/ReluGradReluGradL-?2E??!L-?2E??"@
"gradient_tape/model/dense_1/MatMulMatMul??@X??!?s8;e-??0"2
model/dense_1/MatMulMatMul??????!!?ڴ???0"@
$gradient_tape/model/dense_1/MatMul_1MatMul?u?{:??!?~??Sj??"0
model/dense/BiasAddBiasAddÊ??ϵ?!3??L???"*
model/dense/ReluReluS?%??>??!d	&????">
 gradient_tape/model/dense/MatMulMatMul?]2L?S??!@/??>\??0"N
-gradient_tape/model/dense/BiasAdd/BiasAddGradBiasAddGradvN?????!G޳I??"-
IteratorGetNext/_1_Send??d\??!?ɯl??"0
model/dense/MatMulMatMul?(?X_???!a?9?????0Q      Y@Y?18??5@a??18?S@q[20??@yT?%??>??"?
both?Your program is MODERATELY input-bound because 6.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?48.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t34.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 