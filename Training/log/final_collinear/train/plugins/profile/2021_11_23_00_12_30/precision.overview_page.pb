?	????=@????=@!????=@	?
?ڐ????
?ڐ???!?
?ڐ???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL????=@$D???8@1???U???AJ???`??I?w?J@Y??(]z??rEagerKernelExecute 0*	x?&1?n@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????????!?ـ??X?@)????????1?ـ??X?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty?Z??K??!Q8??]7@)p}Xo?
??1?F֡?5@:Preprocessing2U
Iterator::Model::ParallelMapV2??gB?Ģ?!?8R&<?-@)??gB?Ģ?1?8R&<?-@:Preprocessing2F
Iterator::Model?I?%r???!??&??R<@)?	Q???1??߂?*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate"r?z?f??!ݬ¢?BD@)??zi? ??1????X"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipv???_w??!_W6H?Q@)?o????1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap;7m?i???!fǷ?)E@)?Q,??r?1.q??B???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorK?P?r?!?????)K?P?r?1?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 84.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?
?ڐ???I???bW@QZ?p???@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	$D???8@$D???8@!$D???8@      ??!       "	???U??????U???!???U???*      ??!       2	J???`??J???`??!J???`??:	?w?J@?w?J@!?w?J@B      ??!       J	??(]z????(]z??!??(]z??R      ??!       Z	??(]z????(]z??!??(]z??b      ??!       JGPUY?
?ڐ???b q???bW@yZ?p???@?"B
&gradient_tape/model_1/dense_4/MatMul_1MatMulJtP$#???!JtP$#???"4
model_1/dense_4/MatMulMatMulLze????!?A?D?<??0"B
$gradient_tape/model_1/dense_4/MatMulMatMul?>#?????!?p{???0"D
&gradient_tape/model_1/dense_3/ReluGradReluGradMG??????!???H???"4
model_1/dense_3/BiasAddBiasAddX??+???!`????A??".
model_1/dense_3/ReluRelu??5Y??!Q?T????"D
&gradient_tape/model_1/dense_4/ReluGradReluGradL?^G??!D\4????"B
$gradient_tape/model_1/dense_3/MatMulMatMul^,?.???!u???t??0"R
1gradient_tape/model_1/dense_3/BiasAdd/BiasAddGradBiasAddGrad??=@9S??!)???#O??"-
IteratorGetNext/_1_Send$?:?̋??!?{\4???Q      Y@Y?18??5@a??18?S@q??&/?2@y^,?.???"?
both?Your program is POTENTIALLY input-bound because 84.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?18.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 