?	}w+{@}w+{@!}w+{@	0Z??[ @0Z??[ @!0Z??[ @"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL}w+{@פ???	@1??;?????A6X8I?Ǥ?Ie??E?@Y??++MJ??rEagerKernelExecute 0*	sh??|?r@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicet?p??[??!4?%>8I@)t?p??[??14?%>8I@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??	?_???!??????-@)D??]L3??1pwC?~?+@:Preprocessing2U
Iterator::Model::ParallelMapV22?CP5??!e?????'@)2?CP5??1e?????'@:Preprocessing2F
Iterator::ModelZ??mē??!?,k?#?6@)?B ?8???1??2ܮ&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateN??}????!P?0?\?L@)׆?q?&??1?q??@@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZippA?,_???!?4?wFS@)?@j'??1x??n?)@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6ɏ?k??!dL?H?4M@)???]/Mq?1~B-0???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorH?3?9Ak?!?1??????)H?3?9Ak?1?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?37.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t43.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.90Z??[ @IV?+?->T@Q9-H???%@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	פ???	@פ???	@!פ???	@      ??!       "	??;???????;?????!??;?????*      ??!       2	6X8I?Ǥ?6X8I?Ǥ?!6X8I?Ǥ?:	e??E?@e??E?@!e??E?@B      ??!       J	??++MJ????++MJ??!??++MJ??R      ??!       Z	??++MJ????++MJ??!??++MJ??b      ??!       JGPUY0Z??[ @b qV?+?->T@y9-H???%@?"2
model/dense_1/MatMulMatMulo?J?UU??!o?J?UU??0"@
$gradient_tape/model/dense_1/MatMul_1MatMulFv㘥L??!????????"@
"gradient_tape/model/dense_1/MatMulMatMul浶??D??!???r???0"@
"gradient_tape/model/dense/ReluGradReluGrad?d>?4??!??$?3??"0
model/dense/BiasAddBiasAdd?&?]8O??!?׍???"*
model/dense/ReluRelu?_I?W???!	u??rB??">
 gradient_tape/model/dense/MatMulMatMul0?<mF???!l??,g???0"-
IteratorGetNext/_1_Send?A)?????!?Ӷ??5??"N
-gradient_tape/model/dense/BiasAdd/BiasAddGradBiasAddGrad?d>?4??!)??y??"0
model/dense/MatMulMatMul ?}?.*??!k????{??0Q      Y@Y"e????4@a???E??S@qa???
!@yp?pB??"?
both?Your program is MODERATELY input-bound because 8.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?37.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t43.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 