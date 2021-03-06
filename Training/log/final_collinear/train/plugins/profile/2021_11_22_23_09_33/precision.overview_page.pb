?	)A?G?9@)A?G?9@!)A?G?9@	????????????????!????????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL)A?G?9@??c??S6@1???x#??A?7L4H???I zR&5???Y??$>w???rEagerKernelExecute 0*	V-?j@2U
Iterator::Model::ParallelMapV2\??J?H??!??!1???@)\??J?H??1??!1???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea??*??!O???,5@)a??*??1O???,5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?,&6צ?!??0<?4@)H?ξ? ??1???a3@:Preprocessing2F
Iterator::Model??Ry=??!Kc??;F@)?4???қ?1ӽHK??)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????έ?!?'?F?V;@)@??T???1AJ&???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????E??!???TN?K@)???aڇ?1 FA???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??լ3???!Y?8?e=@)??<?n?1?w?J?f??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor#?tu?bk?!;?SS??)#?tu?bk?1;?SS??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????????I?I?f??V@Q?????/@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??c??S6@??c??S6@!??c??S6@      ??!       "	???x#?????x#??!???x#??*      ??!       2	?7L4H????7L4H???!?7L4H???:	 zR&5??? zR&5???! zR&5???B      ??!       J	??$>w?????$>w???!??$>w???R      ??!       Z	??$>w?????$>w???!??$>w???b      ??!       JGPUY????????b q?I?f??V@y?????/@?"B
&gradient_tape/model_1/dense_4/MatMul_1MatMul??	e???!??	e???"4
model_1/dense_4/MatMulMatMul
???q??!???ܰ??0"B
$gradient_tape/model_1/dense_4/MatMulMatMul?V1????!???z????0"D
&gradient_tape/model_1/dense_3/ReluGradReluGrad?%??????!???w????"4
model_1/dense_3/BiasAddBiasAdd???\ɤ?!Y???!??".
model_1/dense_3/ReluRelu?p?7??!g?$Qf??"D
&gradient_tape/model_1/dense_4/ReluGradReluGrad8r??*??!?޵y?h??"B
$gradient_tape/model_1/dense_3/MatMulMatMuld?P?J??!5e(?K??0"-
IteratorGetNext/_1_Send&?H"N???!??:????"R
1gradient_tape/model_1/dense_3/BiasAdd/BiasAddGradBiasAddGrad&<??B???!w??
????Q      Y@Y?18??5@a??18?S@qa^i?=5@yuS˷???"?
both?Your program is POTENTIALLY input-bound because 86.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?21.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 