?	?h??i?:@?h??i?:@!?h??i?:@	????c??????c??!????c??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?h??i?:@}?;l"?6@1?2?}ƅ??A=?ЕT??I????u???YJ{?/L???rEagerKernelExecute 0*	?|?5^?f@2U
Iterator::Model::ParallelMapV2?o%;6??!@?qHB@)?o%;6??1@?qHB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ٕ???! ?6&??2@)/???0??1??f4??1@:Preprocessing2F
Iterator::ModelLP÷?n??!??+PCJ@)A?w?鱝?1T9??y?/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?&????!P?=??+@)?&????1P?=??+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0???"??!k&????G@)??X32ȍ?1????? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?%Tpx??!?bLh??2@)!?Ky ??1Q??&'|@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????!??!??o??4@)?+=)?j?1??[u????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?K??T?h?!]????)?K??T?h?1]????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 85.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????c??I?E????V@QM_F?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	}?;l"?6@}?;l"?6@!}?;l"?6@      ??!       "	?2?}ƅ???2?}ƅ??!?2?}ƅ??*      ??!       2	=?ЕT??=?ЕT??!=?ЕT??:	????u???????u???!????u???B      ??!       J	J{?/L???J{?/L???!J{?/L???R      ??!       Z	J{?/L???J{?/L???!J{?/L???b      ??!       JGPUY????c??b q?E????V@yM_F?@?"B
&gradient_tape/model_1/dense_4/MatMul_1MatMul??J???!??J???"4
model_1/dense_4/MatMulMatMul?]???S??!??}?????0"B
$gradient_tape/model_1/dense_4/MatMulMatMul??Y?ի??!?0>????0"D
&gradient_tape/model_1/dense_3/ReluGradReluGradR?As?J??!TJI?????"4
model_1/dense_3/BiasAddBiasAddF??t???!?9?l???".
model_1/dense_3/ReluRelu??ߠУ?!?8?z?D??"D
&gradient_tape/model_1/dense_4/ReluGradReluGrada??????!:2vjF??"B
$gradient_tape/model_1/dense_3/MatMulMatMul?????!????"??0"R
1gradient_tape/model_1/dense_3/BiasAdd/BiasAddGradBiasAddGrad?*?KfE??!Xr?????".
model_1/dense_4/ReluRelush?_P???!??<,???Q      Y@Y?18??5@a??18?S@q????D6@yZ??bm???"?
both?Your program is POTENTIALLY input-bound because 85.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?22.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 