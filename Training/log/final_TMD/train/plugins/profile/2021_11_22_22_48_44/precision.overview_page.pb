?	?z6?N@?z6?N@!?z6?N@	H???

(@H???

(@!H???

(@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?z6?N@?????@1?~?f???A??8?Վ??I??ݒp@Y??+,8??rEagerKernelExecute 0*	ʡE??r@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???]/M??!?N?T?<@)???]/M??1?N?T?<@:Preprocessing2U
Iterator::Model::ParallelMapV2)狽_??!?}^?_#6@))狽_??1?}^?_#6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatenm?y????!?????F@)?{,GȨ?1???P??0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?z?΅??!?X???A1@)?? >????1-P_l?&0@:Preprocessing2F
Iterator::Model???<????!???3W@@)@k~??E??1y??n??#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS?Z?!??!(f??P@)+??O8???13ߞ??b@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap1?Zd??!??਄G@)mXSYvq?12??????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?_?5?!j?!i???1???)?_?5?!j?1i???1???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 12.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?41.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t34.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9H???

(@I?,??J)S@Q?????&@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????@?????@!?????@      ??!       "	?~?f????~?f???!?~?f???*      ??!       2	??8?Վ????8?Վ??!??8?Վ??:	??ݒp@??ݒp@!??ݒp@B      ??!       J	??+,8????+,8??!??+,8??R      ??!       Z	??+,8????+,8??!??+,8??b      ??!       JGPUYH???

(@b q?,??J)S@y?????&@?"2
model/dense_1/MatMulMatMul??a????!??a????0"@
$gradient_tape/model/dense_1/MatMul_1MatMul=?BZ???!Z??~a??"@
"gradient_tape/model/dense_1/MatMulMatMulD??Z???!?!?U???0"@
"gradient_tape/model/dense/ReluGradReluGradh"?J㴳?!?ͻ????"0
model/dense/BiasAddBiasAdd?Q:+???!!@#t7??"*
model/dense/ReluRelu?d?????!Y??)????"P
/gradient_tape/model/dense_2/BiasAdd/BiasAddGradBiasAddGrad"?8?M??!?V??d??">
 gradient_tape/model/dense/MatMulMatMul??Όݥ??!?#?????0"-
IteratorGetNext/_1_Send>W,ASz??!z}6
?F??"N
-gradient_tape/model/dense/BiasAdd/BiasAddGradBiasAddGradh"?J㴣?!???>ف??Q      Y@Y?18??5@a??18?S@q?:?.?@y+W???"?
both?Your program is MODERATELY input-bound because 12.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?41.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t34.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 