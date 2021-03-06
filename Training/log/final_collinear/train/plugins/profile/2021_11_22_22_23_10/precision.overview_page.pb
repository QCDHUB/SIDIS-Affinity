?	?x??=@?x??=@!?x??=@	?t???t??!?t??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?x??=@????SG8@1ol?`q??AmU?Y??I?};??@Y???0`???rEagerKernelExecute 0*	??v??j@2U
Iterator::Model::ParallelMapV2??y7??!ؼ??#	A@)??y7??1ؼ??#	A@:Preprocessing2F
Iterator::Modelu???!?#??+L@)????ϧ?1???H"E6@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?g\W̠?!?ӈ??k/@)?g\W̠?1?ӈ??k/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?b??Ր??!??[ϓ?.@)0e?????1Q??8??,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????Z???!a?F??7@)???'???1}?	|??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?8GW??!1?? K?E@)M?^?iN~?1??%X@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapq:?V?S??!??yG?8@)vk???i?1J?z?E??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??lXSYd?!/?T???)??lXSYd?1/?T???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 83.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?t??I??B?[W@Q3I,VA-@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????SG8@????SG8@!????SG8@      ??!       "	ol?`q??ol?`q??!ol?`q??*      ??!       2	mU?Y??mU?Y??!mU?Y??:	?};??@?};??@!?};??@B      ??!       J	???0`??????0`???!???0`???R      ??!       Z	???0`??????0`???!???0`???b      ??!       JGPUY?t??b q??B?[W@y3I,VA-@?"B
&gradient_tape/model_1/dense_4/MatMul_1MatMul??u????!??u????"4
model_1/dense_4/MatMulMatMul?6?"????!????-??0"B
$gradient_tape/model_1/dense_4/MatMulMatMul?TT???!;. ???0"D
&gradient_tape/model_1/dense_3/ReluGradReluGradR?x?Nӭ?!???????"4
model_1/dense_3/BiasAddBiasAdd??!?I٤?!?ؑ??2??".
model_1/dense_3/ReluRelu???4E??!????v??"D
&gradient_tape/model_1/dense_4/ReluGradReluGrad?-;`a???!\???~??"B
$gradient_tape/model_1/dense_3/MatMulMatMulo??fb̛?!??M]??0"R
1gradient_tape/model_1/dense_3/BiasAdd/BiasAddGradBiasAddGradw???8??!3А7??".
model_1/dense_4/ReluRelu?g?E*??!p?<`???Q      Y@Y?18??5@a??18?S@q???w?%9@yh???3???"?
both?Your program is POTENTIALLY input-bound because 83.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?25.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 