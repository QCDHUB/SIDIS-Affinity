?	?D???2%@?D???2%@!?D???2%@	??j???@??j???@!??j???@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?D???2%@??-I?@1]??t??A?mUٯ?I6Z?P?@Y'1?Z??rEagerKernelExecute 0*	??v???k@2U
Iterator::Model::ParallelMapV2?Ù_???!?.???6@)?Ù_???1?.???6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat5z5@i???!vҥc6@)?+ٱ??1,??? 5@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceta?????!?d??.3@)ta?????1?d??.3@:Preprocessing2F
Iterator::Model+?C3O???!??8?B@)?0??Z??1آ.?xI.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?q5?+-??!W?A~??@@)0??L?^??1(\??^?,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?0DN_???!??J??O@)P?R)v??1??'
??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?,??;??!?>3ʧA@)?dq???p?1??N?փ??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorL??pvki?!??P.??)L??pvki?1??P.??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?67.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??j???@I?P??8V@QB??k?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??-I?@??-I?@!??-I?@      ??!       "	]??t??]??t??!]??t??*      ??!       2	?mUٯ??mUٯ?!?mUٯ?:	6Z?P?@6Z?P?@!6Z?P?@B      ??!       J	'1?Z??'1?Z??!'1?Z??R      ??!       Z	'1?Z??'1?Z??!'1?Z??b      ??!       JGPUY??j???@b q?P??8V@yB??k?@?"2
model/dense_1/MatMulMatMul???9???!???9???0"@
$gradient_tape/model/dense_1/MatMul_1MatMul??`????!C?S5U??"@
"gradient_tape/model/dense_1/MatMulMatMul?Q??zG??!굈Ky<??0"@
"gradient_tape/model/dense/ReluGradReluGrada??3?6??!B??4J??"0
model/dense/BiasAddBiasAdd?.?????!?*??"*
model/dense/ReluRelu
=?P????!\???d??">
 gradient_tape/model/dense/MatMulMatMul??2?A??!*)?S????0"-
IteratorGetNext/_1_Send*?+?)??!Mz??X[??"N
-gradient_tape/model/dense/BiasAdd/BiasAddGradBiasAddGrad?[?????!.9?0????"0
model/dense/MatMulMatMul?.?????!?z?1????0Q      Y@Y?18??5@a??18?S@q?է??A??y??? ?D??"?

both?Your program is POTENTIALLY input-bound because 20.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?67.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 