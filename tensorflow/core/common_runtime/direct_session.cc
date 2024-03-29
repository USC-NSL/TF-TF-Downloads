/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/direct_session.h"

#include <atomic>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/simple_placer.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph.pb_text.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_tracer.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

namespace {

auto* direct_session_runs = monitoring::Counter<0>::New(
    "/tensorflow/core/direct_session_runs",
    "The number of times DirectSession::Run() has been called.");

int32 NumInterOpThreadsFromSessionOptions(const SessionOptions& options) {
  const int32 t = options.config.inter_op_parallelism_threads();

  // Yitao-TLS-Begin
  // The default num of thread is 12, let's force to set it as 1024
  return 9999;
  // Yitao-TLS-End

  if (t != 0) return t;
  // Default to using the number of cores available in the process.
  return port::NumSchedulableCPUs();
}

thread::ThreadPool* NewThreadPoolFromSessionOptions(
    const SessionOptions& options) {
  const int32 num_threads = NumInterOpThreadsFromSessionOptions(options);
  VLOG(1) << "Direct session inter op parallelism threads: " << num_threads;
  return new thread::ThreadPool(options.env, "Compute", num_threads);
}

thread::ThreadPool* NewThreadPoolFromThreadPoolOptions(
    const SessionOptions& options,
    const ThreadPoolOptionProto& thread_pool_options, int pool_number) {
  int32 num_threads = thread_pool_options.num_threads();
  if (num_threads == 0) {
    num_threads = NumInterOpThreadsFromSessionOptions(options);
  }
  VLOG(1) << "Direct session inter op parallelism threads for pool "
          << pool_number << ": " << num_threads;
  return new thread::ThreadPool(
      options.env, strings::StrCat("Compute", pool_number), num_threads);
}

thread::ThreadPool* GlobalThreadPool(const SessionOptions& options) {
  static thread::ThreadPool* const thread_pool =
      NewThreadPoolFromSessionOptions(options);
  return thread_pool;
}

// TODO(vrv): Figure out how to unify the many different functions
// that generate RendezvousKey, since many of them have to be
// consistent with each other.
string GetRendezvousKey(const string& tensor_name,
                        const DeviceAttributes& device_info,
                        const FrameAndIter& frame_iter) {
  return strings::StrCat(device_info.name(), ";",
                         strings::FpToString(device_info.incarnation()), ";",
                         device_info.name(), ";", tensor_name, ";",
                         frame_iter.frame_id, ":", frame_iter.iter_id);
}

}  // namespace

// // Yitao-TLS-Begin
// // The TLS scheduler is using a token (here, next_run_id) to decide which Session object
// // should run its own Session.run() next. Every time one Session object's Session.run() 
// // finished, it will set next_run_id = -1 to let TLS scheduler decide who should run next.
// // Yitao-to-do: currently, the TLS scheduler has the following limitations:
// //              (1) it is hard-coded to 1:1 ratio between two Session objects. So if one Session
// //                  object has five Session.run()s, and the other has ten Session.run()s. Then the
// //                  second one will be stuck infinitly. So I need to update it to make it smarter...
// //              (2) the current token-based scheduling should be more flexible. Namely, I need to
// //                  make this TLS_scheduler() module more flexbile to support different scheduling policies
// void TLS_scheduler(std::mutex* sched_lock, std::condition_variable* sched_cv, int* next_run_id, bool* someone_running, std::priority_queue<int, std::vector<int>, std::greater<int>>* wait_queue) {
//   LOG(INFO) << "[Yitao] ****** TLS(), we are starting the TLS Scheduler!!! ******";
//   int my_id = -1;
//   int counter = -1;

//   // while (true) {
//   //   std::unique_lock<std::mutex> lk(*sched_lock);
//   //   sched_cv->wait(lk, [my_id, next_run_id](){return *next_run_id == my_id;});

//   //   counter = (counter + 1) % 2;
//   //   *next_run_id = counter;

//   //   LOG(INFO) << "[Yitao] ****** TLS Scheduler decided to run next_run_id = " << *next_run_id;

//   //   sched_cv->notify_all();
//   // }

//   while (true) {
//     std::unique_lock<std::mutex> lk(*sched_lock);
//     sched_cv->wait(lk, [my_id, next_run_id](){return *next_run_id == my_id;});

//     *someone_running = false;

//     if ((*wait_queue).empty()) {
//       LOG(INFO) << "[Yitao] ****** wait_queue is empty...";
//       *next_run_id = 0; // weird bug here, if default is 1 and send 1000 mnist concurrently.
//     } else {
//       LOG(INFO) << "[Yitao] ****** wait_queue has " << (*wait_queue).size() << " nodes left before poping";
//       *next_run_id = (*wait_queue).top();
//       (*wait_queue).pop();
//       if (!(*wait_queue).empty() && (*next_run_id) == 0) {
//         LOG(INFO) << "[Yitao] Duangduangduang, TLS is working!!!";
//       }
//     }

//     // LOG(INFO) << "[Yitao] ****** TLS Scheduler decided to run next_run_id = " << *next_run_id;

//     sched_cv->notify_all();
//   }
// }
// // Yitao-TLS-End

// // Yitao-TLS-Begin
// // The thread for TLS scheduler to serve as a centralized scheduler
// void TLS_scheduler(std::priority_queue<sessRunInfo>* TLS_queue, std::mutex* sched_lock, std::condition_variable* TLS_cv, std::condition_variable* sched_cv, int* next_sess_id, int* next_sess_run_id, bool* notify_done) {
//   while (true) {
//     std::unique_lock<std::mutex> lk(*sched_lock);

//     // In Process(), each node will notify TLS_cv twice,
//     // and as long as TLS_queue is not empty, this thread will be awaken from suspending
//     TLS_cv->wait(lk, [TLS_queue](){return !TLS_queue->empty();});

//     sessRunInfo mySessRunInfo = TLS_queue->top();
//     TLS_queue->pop();

//     *next_sess_id = mySessRunInfo.mySessId;
//     *next_sess_run_id = mySessRunInfo.mySessRunId;

//     // LOG(INFO) << "[TLS scheduler] decided to run sess_id = " << *next_sess_id << ", sess_run_id = " << *next_sess_run_id << ", and after pop, TLS_queue->size() = " << TLS_queue->size();

//     // notify_done is used to ensure that this (next_sess_id, next_sess_run_id)
//     // can guarantee the corresponding Sess.run() will be awaken from suspending.
//     // notify_done is used to fix the previous threading bug!!!
//     *notify_done = false;

//     while (!*notify_done) {
//       lk.unlock(); // need to unlock lk, otherwise will lead to contension
//       sched_cv->notify_all();
//       lk.lock();
//     }
//   }
// }
// // Yitao-TLS-End

class DirectSessionFactory : public SessionFactory {
 public:
  DirectSessionFactory() {}

  // Yitao-TLS-Begin
  // ~DirectSessionFactory() {my_thread->join();}
  ~DirectSessionFactory() {}
  // Yitao-TLS-End

  bool AcceptsOptions(const SessionOptions& options) override {
    return options.target.empty();
  }

  Session* NewSession(const SessionOptions& options) override {

    // Yitao-TLS-Begin
    {
      mutex_lock l(sessions_lock_);
      if (first_constructor_called) {
        first_constructor_called = false;
        LOG(INFO) << "[Yitao] Testing: DirectSessionFactory::DirectSessionFactory(), we are calling initialization function of DirectSessionFactory @@@@@@";
        sess_count = -1;

        // next_sess_id = new int;
        // next_sess_run_id = new int;
        // *next_sess_id = -1;
        // *next_sess_run_id = -1;

        // notify_done = new bool;
        // *notify_done = false;

        // sched_lock = new std::mutex;
        // TLS_cv = new std::condition_variable;
        // sched_cv = new std::condition_variable;

        // TLS_queue = new std::priority_queue<sessRunInfo>;

        // my_thread = new std::thread(TLS_scheduler, TLS_queue, sched_lock, TLS_cv, sched_cv, next_sess_id, next_sess_run_id, notify_done);
      
        olympia_scheduler = new OlympiaScheduler;
      }
    }
    // Yitao-TLS-End

    // Must do this before the CPU allocator is created.
    if (options.config.graph_options().build_cost_model() > 0) {
      EnableCPUAllocatorFullStats(true);
    }
    std::vector<Device*> devices;
    Status s = DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices);
    if (!s.ok()) {
      LOG(ERROR) << s;
      return nullptr;
    }

    for (int i = 0; i < devices.size(); i++) {
      auto myDevice = devices[i];
      LOG(INFO) << "[Yitao] Test in NewSession: device = " << myDevice->name();
    }

    // Yitao-TLS-Begin
    // update the sess_count variable with lock to make sure it is thread-safe
    int sess_count_value;
    {
      mutex_lock l(sessions_lock_);
      sess_count += 1;
      sess_count_value = sess_count;
    }
    LOG(INFO) << "[Yitao] Testing: DirectSessionFactory::NewSession(), we have " << sess_count_value << " DirectSessions now! @@@@@@";
    // Yitao-TLS-End

    DirectSession* session =
        new DirectSession(options, new DeviceMgr(devices), this, sess_count_value);
    {
      mutex_lock l(sessions_lock_);
      sessions_.push_back(session);
    }

    // Yitao-TLS-Begin
    // *** This idea has been deprecated in the current version,
    //     but might be usable in future versions...
    // We are having the first Session, let's start the TLS scheduler
    // The reason I didn't put this into DirectSessionFactory's constructor function
    // is that the constructor function is also called by the client python script.
    // So if I put the start of TLS scheduler in the contructor function,
    // then we will have multiple TLS scheduler...
    // if (GetSessionCount() == 1) {
      // sched_lock = new std::mutex;
      // LOG(INFO) << "[Yitao] *** we have sched_lock address = " << sched_lock;
      // sched_cv = new std::condition_variable;
      // next_run_id = new int;
      // *next_run_id = 1;
      // my_thread = new std::thread(TLS_scheduler, sched_lock, sched_cv, next_run_id);
    // }
    // Yitao-TLS-End

    return session;
  }

  Status Reset(const SessionOptions& options,
               const std::vector<string>& containers) override {
    std::vector<DirectSession*> sessions_to_reset;
    {
      mutex_lock l(sessions_lock_);
      // We create a copy to ensure that we don't have a deadlock when
      // session->Close calls the DirectSessionFactory.Deregister, which
      // acquires sessions_lock_.
      std::swap(sessions_to_reset, sessions_);
    }
    Status s;
    for (auto session : sessions_to_reset) {
      s.Update(session->Reset(containers));
    }
    // TODO(suharshs): Change the Reset behavior of all SessionFactories so that
    // it doesn't close the sessions?
    for (auto session : sessions_to_reset) {
      s.Update(session->Close());
    }
    return s;
  }

  void Deregister(const DirectSession* session) {
    mutex_lock l(sessions_lock_);
    sessions_.erase(std::remove(sessions_.begin(), sessions_.end(), session),
                    sessions_.end());
  }

  // Yitao-TLS-Begin
  // return the value of sess_count
  int GetSessionCount() {
    return sess_count;
  }

  // int* next_sess_id;
  // int* next_sess_run_id;

  // bool* notify_done;

  // std::mutex* sched_lock; // shared by both TLS_cv and sched_cv
  // std::condition_variable* TLS_cv;
  // std::condition_variable* sched_cv;

  // std::priority_queue<sessRunInfo>* TLS_queue;

  // std::thread* my_thread;

  OlympiaScheduler* olympia_scheduler;

  static bool first_constructor_called;
  // Yitao-TLS-End

 private:
  mutex sessions_lock_;
  std::vector<DirectSession*> sessions_ GUARDED_BY(sessions_lock_);

  // Yitao-TLS-Begin
  // sess_count is used to set sess_id for each Session object.
  // Namely, every time we initialize a Session object, we will
  // use sess_count to assign a unique sess_id to the new Session object
  int sess_count;
  // Yitao-TLS-End

};

bool DirectSessionFactory::first_constructor_called = true;

class DirectSessionRegistrar {
 public:
  DirectSessionRegistrar() {
    SessionFactory::Register("DIRECT_SESSION", new DirectSessionFactory());
  }
};
static DirectSessionRegistrar registrar;

std::atomic_int_fast64_t DirectSession::step_id_counter_(1);

// NOTE: On Android with a single device, there is never
// a risk of an OpKernel blocking indefinitely:
//
// 1) No operations do I/O that depends on other simultaneous kernels,
//
// 2) Recv nodes always complete immediately: The inputs are sent into
//    the local rendezvous before we start the executor, so the
//    corresponding recvs will not block.
//
// Based on these assumptions, we can use the same thread pool for
// both "non-blocking" and "blocking" OpKernels on Android.
//
// This may change down the road when we add support for multiple
// devices that run concurrently, in which case we will need to
// revisit this decision.
void DirectSession::SchedClosure(thread::ThreadPool* pool,
                                 std::function<void()> c) {
// TODO(sanjay): Get rid of __ANDROID__ path
#ifdef __ANDROID__
  // On Android, there is no implementation of ThreadPool that takes
  // std::function, only Closure, which we cannot easily convert.
  //
  // Instead, we just run the function in-line, which is currently
  // safe given the reasoning above.
  c();
#else
  pool->Schedule(std::move(c));
#endif  // __ANDROID__
}

DirectSession::DirectSession(const SessionOptions& options,
                             const DeviceMgr* device_mgr,
                             DirectSessionFactory* const factory,
                             int sess_count = -1) // [Yitao]: here sess_count is a new variable added by myself...
    : options_(options),
      device_mgr_(device_mgr),
      factory_(factory),
      cancellation_manager_(new CancellationManager()),
      operation_timeout_in_ms_(options_.config.operation_timeout_in_ms()) {

  LOG(INFO) << "[Yitao] ****** DirectSession::DirectSession() ******";

  // Yitao-TLS-Begin
  // In the constructor function of DirectSession,
  // we assign a unique sess_id to the new Session object
  // with the help of sess_count.
  sess_id = sess_count;
  LOG(INFO) << "[Yitao] ****** DirectSession::DirectSession(), we have sess_id = " << sess_id;

  // next_sess_id = factory_->next_sess_id;
  // next_sess_run_id = factory_->next_sess_run_id;

  // notify_done = factory_->notify_done;

  // sched_lock = factory_->sched_lock;
  // TLS_cv = factory_->TLS_cv;
  // sched_cv = factory_->sched_cv;

  // TLS_queue = factory_->TLS_queue;

  // might consider put sess_run_count's initialization under sess_run_count_lock
  sess_run_count = -1;

  cost_model_generated = new bool;
  *cost_model_generated = false;

  TLS_cost_model = new std::unordered_map<string, int>;

  olympia_scheduler = factory_->olympia_scheduler;

  // Yitao-TLS-End

  // Yitao-TLS-Begin
  if (options_.config.session_inter_op_thread_pool_size() > 0) {
    LOG(INFO) << "[Yitao] in DirectSession::DirectSession(), case one with options_.config.session_inter_op_thread_pool_size() = " << options_.config.session_inter_op_thread_pool_size(); 
    for (int i = 0; i < options_.config.session_inter_op_thread_pool_size();
         ++i) {
      thread_pools_.push_back(NewThreadPoolFromThreadPoolOptions(
          options_, options_.config.session_inter_op_thread_pool(i), i));
    }
    owns_thread_pools_ = true;
  } else if (options_.config.use_per_session_threads()) {
    LOG(INFO) << "[Yitao] in DirectSession::DirectSession(), case two with options_.config.use_per_session_threads() = " << options_.config.use_per_session_threads();
    thread_pools_.push_back(NewThreadPoolFromSessionOptions(options_));
    owns_thread_pools_ = true;
  } else {
    LOG(INFO) << "[Yitao] in DirectSession::DirectSession(), case three...";
    thread_pools_.push_back(GlobalThreadPool(options));
    owns_thread_pools_ = false;
  }
  // Yitao-TLS-End

  // The default value of sync_on_finish will be flipped soon and this
  // environment variable will be removed as well.
  Status status =
      ReadBoolFromEnvVar("TF_SYNC_ON_FINISH", true, &sync_on_finish_);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
  // NOTE(mrry): We do not need to use a unique string for the session
  // handle, because DirectSession owns its devices. This may change
  // in future versions.
  session_handle_ = "direct";
  int devices_added = 0;
  if (options.config.log_device_placement()) {
    const string mapping_str = device_mgr_->DeviceMappingString();
    if (mapping_str.empty()) {
      printf("Device mapping: no known devices.\n");
    } else {
      printf("Device mapping:\n%s", mapping_str.c_str());
    }
    LOG(INFO) << "Device mapping:\n" << mapping_str;
  }
  for (auto d : device_mgr_->ListDevices()) {
    LOG(INFO) << "[Yitao] Test in DirectSession::DirectSession(), device = " << d->name();
    devices_.push_back(d);
    device_set_.AddDevice(d);
    d->op_segment()->AddHold(session_handle_);

    // The first device added is special: it is the 'client device' (a
    // CPU device) from which we feed and fetch Tensors.
    if (devices_added == 0) {
      device_set_.set_client_device(d);
    }
    ++devices_added;
  }
}

DirectSession::~DirectSession() {
  if (!closed_) Close().IgnoreError();
  for (auto& it : partial_runs_) {
    it.second.reset(nullptr);
  }
  for (auto& it : executors_) {
    it.second.reset();
  }
  for (auto d : device_mgr_->ListDevices()) {
    d->op_segment()->RemoveHold(session_handle_);
  }
  delete cancellation_manager_;
  if (owns_thread_pools_) {
    for (auto* p : thread_pools_) delete p;
  }

  execution_state_.reset(nullptr);
  flib_def_.reset(nullptr);
}

Status DirectSession::MaybeInitializeExecutionState(
    const GraphDef& graph, bool* out_already_initialized) {
  // If already initialized, do nothing.
  if (flib_def_ && execution_state_) {
    *out_already_initialized = true;
    return Status::OK();
  }
  // Set up the per-session execution state.
  // NOTE(mrry): The function library created here will be used for
  // all subsequent extensions of the graph.
  flib_def_.reset(
      new FunctionLibraryDefinition(OpRegistry::Global(), graph.library()));
  SimpleGraphExecutionStateOptions options;
  options.device_set = &device_set_;
  options.session_options = &options_;
  // TODO(mrry,suharshs): We explicitly copy `graph` so that
  // `MakeForBaseGraph()` can take ownership of its
  // contents. Previously this happened implicitly in calls to the
  // `SimpleGraphExecutionState`. Other sessions call
  // `MakeForBaseGraph` in such a way that we can destructively read
  // the passed-in `GraphDef`. In principle we could do the same here,
  // with a wider refactoring; we might revise the direct session so
  // that it copies the graph fewer times.
  GraphDef temp(graph);
  TF_RETURN_IF_ERROR(SimpleGraphExecutionState::MakeForBaseGraph(
      &temp, options, &execution_state_));
  graph_created_ = true;
  *out_already_initialized = false;
  return Status::OK();
}

Status DirectSession::Create(const GraphDef& graph) {
  if (graph.node_size() > 0) {
    mutex_lock l(graph_def_lock_);
    if (graph_created_) {
      return errors::AlreadyExists(
          "A Graph has already been created for this session.");
    }
    return ExtendLocked(graph);
  }
  return Status::OK();
}

Status DirectSession::Extend(const GraphDef& graph) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(graph_def_lock_);
  return ExtendLocked(graph);
}

Status DirectSession::ExtendLocked(const GraphDef& graph) {
  bool already_initialized;
  // If this is the first call, we can initialize the execution state
  // with `graph` and do not need to call `Extend()`.
  TF_RETURN_IF_ERROR(
      MaybeInitializeExecutionState(graph, &already_initialized));
  if (already_initialized) {
    std::unique_ptr<SimpleGraphExecutionState> state;
    TF_RETURN_IF_ERROR(execution_state_->Extend(graph, &state));
    execution_state_.swap(state);
  }
  return Status::OK();
}

Status DirectSession::Run(const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs) {
  RunMetadata run_metadata;
  return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
             &run_metadata);
}

Status DirectSession::CreateDebuggerState(
    const DebugOptions& debug_options, int64 session_run_index,
    int64 executor_step_index, const std::vector<string>& input_names,
    const std::vector<string>& output_names,
    const std::vector<string>& target_names,
    std::unique_ptr<DebuggerStateInterface>* debugger_state) {
  TF_RETURN_IF_ERROR(
      DebuggerStateRegistry::CreateState(debug_options, debugger_state));
  TF_RETURN_IF_ERROR(debugger_state->get()->PublishDebugMetadata(
      debug_options.global_step(), session_run_index, executor_step_index,
      input_names, output_names, target_names));
  return Status::OK();
}

Status DirectSession::DecorateAndPublishGraphForDebug(
    const DebugOptions& debug_options, Graph* graph, Device* device) {
  std::unique_ptr<DebugGraphDecoratorInterface> decorator;
  TF_RETURN_IF_ERROR(
      DebugGraphDecoratorRegistry::CreateDecorator(debug_options, &decorator));

  TF_RETURN_IF_ERROR(decorator->DecorateGraph(graph, device));
  TF_RETURN_IF_ERROR(decorator->PublishGraph(*graph, device->name()));
  return Status::OK();
}

Status DirectSession::Run(const RunOptions& run_options,
                          const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata) {

  // LOG(INFO) << "[Yitao] ****** DirectSession::Run() ******";
  LOG(INFO) << "[Yitao] ****** DirectSession::Run(), we have sess_id = " << sess_id;
  int sess_run_id;
  {
    mutex_lock l(sess_run_count_lock);
    sess_run_count += 1;
    sess_run_id = sess_run_count;
  }
  LOG(INFO) << "[Yitao] ****** DirectSession::Run(), we have sess_run_id = " << sess_run_id;

  int tom_timeout = run_options.timeout_in_ms();
  // LOG(INFO) << "[Yitao] ****** DirectSession::Run(), we have run_options.timeout_in_ms = " << tom_timeout;
  // if (tom_timeout > 70000)
  //   LOG(INFO) << "[Yitao] ... tom_timeout > 70k";
  // else
  //   LOG(INFO) << "[Yitao] ... tom_timeout <= 70k";


  // Yitao-TLS-Begin
  SessRunInfo sr_info = SessRunInfo(sess_id, sess_run_id);

  sr_info.priority = (tom_timeout - 59000) / 10000;
  LOG(INFO) << "[Yitao] ****** DirectSession::Run(), we have run_options.timeout_in_ms = " << tom_timeout << " with sr_info.priority = " << sr_info.priority;


  // std::condition_variable* my_cv = new std::condition_variable;
  int* my_cumulated_cost = new int;
  *my_cumulated_cost = 0;

  int64* last_decision_time = new int64;
  *last_decision_time = 0;

  std::mutex* my_lock = new std::mutex;

  int* cv_check_count;
  cv_check_count = new int;
  *cv_check_count = 0;

  {
    olympia_scheduler->SessRunRegister(sr_info);
  }
  // Yitao-TLS-End

  // // Yitao-TLS-Begin
  // if (true) { // <====== should_we_push_this_node(node) for node level scheduling here
  //   {
  //     // since we are modifying the shared TLS_queue,
  //     // we need sched_lock to protect it.
  //     std::unique_lock<std::mutex> lk(*sched_lock);
  //     TLS_queue->push(sessRunInfo(sess_id, sess_run_id));
  //     LOG(INFO) << "[Process] pushing sess_id = " << sess_id << ", sess_run_id = " << sess_run_id << " to queue! After push, TLS_queue->size = " << TLS_queue->size();
  //   }

  //   // notify TLS_scheduler to schedule the next node in TLS queue
  //   TLS_cv->notify_all();

  //   {
  //     std::unique_lock<std::mutex> lk(*sched_lock);
  //     sched_cv->wait(lk, [sess_run_id, this](){
  //       // print some meta-data for debuging
  //       bool tmp = *next_sess_id == sess_id && *next_sess_run_id == sess_run_id;
  //       LOG(INFO) << "[meta] sess_id = " << sess_id << ", sess_run_id = " << sess_run_id << ", next_sess_id = " << *next_sess_id << ", next_sess_run_id = " << *next_sess_run_id << ((tmp) ? " => true" : " => false");
        
  //       // reset next_sess_id and next_sess_run_id.
  //       // Without doing so, then if next Sess.run() happen to have the same sess_id,
  //       // it will be executed as well as be pushed into the queue, leading to bug
  //       if (tmp) {
  //         *next_sess_id = -1;
  //         *next_sess_run_id = -1;
  //       }
  //       return tmp;
  //     });
  //     // If we reach this point, TLS_scheduler's notify_all() has worked,
  //     // so we can stop TLS_scheduler's while loop for notify_all().
  //     *notify_done = true;
  //   }
  // }
  // // Yitao-TLS-End

  TF_RETURN_IF_ERROR(CheckNotClosed());
  direct_session_runs->GetCell()->IncrementBy(1);
  {
    mutex_lock l(graph_def_lock_);
    if (!graph_created_) {
      return errors::InvalidArgument(
          "Session was not created with a graph before Run()!");
    }
  }

  // Extract the inputs names for this run of the session.
  std::vector<string> input_tensor_names;
  input_tensor_names.reserve(inputs.size());
  for (const auto& it : inputs) {
    input_tensor_names.push_back(it.first);
  }

  if (run_options.inter_op_thread_pool() < 0 ||
      run_options.inter_op_thread_pool() >= thread_pools_.size()) {
    return errors::InvalidArgument("Invalid inter_op_thread_pool: ",
                                   run_options.inter_op_thread_pool());
  }
  thread::ThreadPool* pool = thread_pools_[run_options.inter_op_thread_pool()];

  // // Yitao-TLS-Begin
  // LOG(INFO) << "[Yitao] in DirectSession::Run(), run_options.inter_op_thread_pool() = " << run_options.inter_op_thread_pool();
  // LOG(INFO) << "[Yitao] in DirectSession::Run(), pool->NumThreads() = " << pool->NumThreads();
  // // Yitao-TLS-End

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;
  RunStateArgs run_state_args(run_options.debug_options());

  Executor::Args args;
  args.step_id = step_id_counter_.fetch_add(1);

  TF_RETURN_IF_ERROR(
      GetOrCreateExecutors(pool, input_tensor_names, output_names, target_nodes,
                           &executors_and_keys, &run_state_args));
  const int64 executor_step_count = executors_and_keys->step_count.fetch_add(1);

  std::unique_ptr<DebuggerStateInterface> debugger_state;
  if (!run_options.debug_options().debug_tensor_watch_opts().empty()) {
    TF_RETURN_IF_ERROR(CreateDebuggerState(
        run_options.debug_options(), args.step_id, executor_step_count,
        input_tensor_names, output_names, target_nodes, &debugger_state));
  }

  // Configure a call frame for the step, which we use to feed and
  // fetch values to and from the executors.
  FunctionCallFrame call_frame(executors_and_keys->input_types,
                               executors_and_keys->output_types);
  gtl::InlinedVector<Tensor, 4> feed_args(inputs.size());
  for (const auto& it : inputs) {
    if (it.second.dtype() == DT_RESOURCE) {
      Tensor tensor_from_handle;
      TF_RETURN_IF_ERROR(
          ResourceHandleToInputTensor(it.second, &tensor_from_handle));
      feed_args[executors_and_keys->input_name_to_index[it.first]] =
          tensor_from_handle;
    } else {
      feed_args[executors_and_keys->input_name_to_index[it.first]] = it.second;
    }
  }
  Status s = call_frame.SetArgs(feed_args);
  if (errors::IsInternal(s)) {
    return errors::InvalidArgument(s.error_message());
  } else if (!s.ok()) {
    return s;
  }

  // Create a run state and start execution.
  RunState run_state(args.step_id, &devices_);
  run_state.rendez = new IntraProcessRendezvous(device_mgr_.get());
  CancellationManager step_cancellation_manager;
  args.call_frame = &call_frame;

  // Start parallel Executors.
  const size_t num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state.rendez, [&run_state](const Status& ret) {
        {
          mutex_lock l(run_state.mu_);
          run_state.status.Update(ret);
        }
        run_state.executors_done.Notify();
      });

  args.rendezvous = run_state.rendez;
  args.cancellation_manager = &step_cancellation_manager;
  args.runner = [this, pool](Executor::Args::Closure c) {
    SchedClosure(pool, std::move(c));
  };
  args.session_state = &session_state_;
  args.tensor_store = &run_state.tensor_store;
  args.step_container = &run_state.step_container;
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, run_state_args.handle);
  }
  args.sync_on_finish = sync_on_finish_;

  const bool do_trace = (run_options.trace_level() > RunOptions::NO_TRACE);

  bool update_cost_model = false;
  if (options_.config.graph_options().build_cost_model() > 0) {
    const int64 build_cost_model_every =
        options_.config.graph_options().build_cost_model();
    const int64 build_cost_model_after =
        options_.config.graph_options().build_cost_model_after();
    int64 measure_step_count = executor_step_count - build_cost_model_after;
    if (measure_step_count >= 0) {
      update_cost_model =
          ((measure_step_count + 1) % build_cost_model_every == 0);
    }
  }

  // Yitao-TLS-Begin
  // simpliy set update_cost_model as true
  // sess_run_id = 0 and 1 is for model loading
  // sess_run_id = 2 will have much longer overhead
  // so we pick sess_run_id = 5, just for fun...
  int sess_run_threshold = 5;
  bool force_trace_and_update_cost_model = false;
  if (sess_run_id == sess_run_threshold) {
    update_cost_model = true;
    force_trace_and_update_cost_model = true;
  }
  // Yitao-TLS-End

  if (do_trace || update_cost_model) {
    run_state.collector.reset(
        new StepStatsCollector(run_metadata->mutable_step_stats()));
    args.stats_collector = run_state.collector.get();
  }

#if GOOGLE_CUDA

  // LOG(INFO) << "[Yitao] @@@@@@ GOOGLE_CUDA = true @@@@@@";

  std::unique_ptr<GPUTracer> tracer;
  // if (run_options.trace_level() >= RunOptions::HARDWARE_TRACE) {
  // if (force_trace_and_update_cost_model && sess_run_id >= sess_run_threshold) {   // <======================= Pay attention
  if (force_trace_and_update_cost_model || run_options.trace_level() >= RunOptions::HARDWARE_TRACE) {  // <======================= Pay attention

    LOG(INFO) << "[Yitao] @@@@@@ Yes, we are starting GPU Tracer! @@@@@@";

    tracer.reset(CreateGPUTracer());
    // tracer will be NULL on non-GPU platforms.
    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
    if (tracer) tracer->Start().IgnoreError();
  }
#endif  // GOOGLE_CUDA

  // Register this step with session's cancellation manager, so that
  // `Session::Close()` will cancel the step.
  CancellationToken cancellation_token =
      cancellation_manager_->get_cancellation_token();
  bool already_cancelled = !cancellation_manager_->RegisterCallback(
      cancellation_token, [&step_cancellation_manager]() {
        step_cancellation_manager.StartCancel();
      });
  if (already_cancelled) {
    // NOTE(mrry): If we don't explicitly notify
    // `run_state.executors_done`, the RunState destructor would
    // block on this notification.
    run_state.executors_done.Notify();
    delete barrier;
    return errors::Cancelled("Run call was cancelled");
  }

  // Yitao-TLS-Begin
  // args.sess_id = sess_id;

  // args.next_sess_id = next_sess_id;
  // args.next_sess_run_id = next_sess_run_id;

  // args.notify_done = notify_done;

  // args.sched_lock = sched_lock;
  // args.TLS_cv = TLS_cv;
  // args.sched_cv = sched_cv;

  // args.TLS_queue = TLS_queue;

  // args.sess_run_id = sess_run_id;

  // *** Per Session
  args.cost_model_generated = cost_model_generated;
  args.TLS_cost_model = TLS_cost_model;
  args.olympia_scheduler = olympia_scheduler;

  // *** Per Session::Run
  args.cv_check_count = cv_check_count;
  args.sr_info = sr_info;
  // args.my_cv = my_cv;
  args.my_cumulated_cost = my_cumulated_cost;
  args.last_decision_time = last_decision_time;
  args.my_lock = my_lock;

  LOG(INFO) << "[Yitao] There are " << num_executors << " Executors in executors_and_keys...";
  // Yitao-TLS-End

  for (const auto& item : executors_and_keys->items) {
    item.executor->PrintDeviceInfo();
    item.executor->RunAsync(args, barrier->Get());
  }

  WaitForNotification(&run_state, &step_cancellation_manager,
                      run_options.timeout_in_ms() > 0
                          ? run_options.timeout_in_ms()
                          : operation_timeout_in_ms_);

  if (!cancellation_manager_->DeregisterCallback(cancellation_token)) {
    // The step has been cancelled: make sure we don't attempt to receive the
    // outputs as this would make it block forever.
    mutex_lock l(run_state.mu_);
    run_state.status.Update(errors::Cancelled("Run call was cancelled"));
  }

#if GOOGLE_CUDA
  if (tracer) {
    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!

    LOG(INFO) << "[Yitao] @@@@@@ Yes, we are stopping GPU Tracer! @@@@@@";

    tracer->Stop().IgnoreError();
    tracer->Collect(args.stats_collector).IgnoreError();
  }
#endif  // GOOGLE_CUDA

  {
    mutex_lock l(run_state.mu_);
    TF_RETURN_IF_ERROR(run_state.status);
  }

  // Receive outputs.
  if (outputs) {
    std::vector<Tensor> sorted_outputs;
    Status s = call_frame.ConsumeRetvals(&sorted_outputs);
    if (errors::IsInternal(s)) {
      return errors::InvalidArgument(s.error_message());
    } else if (!s.ok()) {
      return s;
    }
    outputs->clear();
    outputs->reserve(sorted_outputs.size());
    for (const string& output_name : output_names) {
      outputs->emplace_back(
          std::move(sorted_outputs[executors_and_keys
                                       ->output_name_to_index[output_name]]));
    }
  }

  // Save the output tensors of this run we choose to keep.
  TF_RETURN_IF_ERROR(
      run_state.tensor_store.SaveTensors(output_names, &session_state_));

  // Build and return the cost model as instructed.
  mutex_lock l(executor_lock_);
  if (update_cost_model) {
    // Build the cost model
    std::unordered_map<string, const Graph*> device_to_graph;
    for (const PerPartitionExecutorsAndLib& partition :
         executors_and_keys->items) {
      const Graph* graph = partition.graph;
      const string device = partition.flib->device()->name();
      device_to_graph[device] = graph;

      LOG(INFO) << "[Yitao] we have device: " << device;
    }
    args.stats_collector->BuildCostModel(&cost_model_manager_, device_to_graph);

    // annotate stats onto cost graph.
    CostGraphDef* cost_graph = run_metadata->mutable_cost_graph();
    for (const auto& item : executors_and_keys->items) {
      TF_RETURN_IF_ERROR(
          cost_model_manager_.AddToCostGraphDef(item.graph, cost_graph));
    }

    // // Yitao-TLS-Begin
    // for (int i = 0; i < cost_graph->node_size(); i++) {
    //   const auto& myNode = cost_graph->node(i);
    //   LOG(INFO) << "[Yitao] CostModel node " << myNode.id() << "-" << myNode.name() << "-" << myNode.device() << "-" << myNode.compute_cost();
    // }
    // // Yitao-TLS-End

    // Yitao-TLS-Begin
    for (int i = 0; i < cost_graph->node_size(); i++) {
      const auto& myNode = cost_graph->node(i);
      if (myNode.compute_cost() >= 0) {                                   // <=== pay attention
        TLS_cost_model->emplace(myNode.name(), myNode.compute_cost());
        // LOG(INFO) << "[Yitao] we are recording Node " << myNode.id() << " " << myNode.name() << " with cost of " << myNode.compute_cost() << " on device " << myNode.device() << ", and (" << myNode.temporary_memory_size() << ", " << myNode.host_temp_memory_size() << ", " << myNode.device_temp_memory_size() << ", " << myNode.host_persistent_memory_size() << ", " << myNode.device_persistent_memory_size() << ") with output_info_size = " << myNode.output_info_size();
        // for (int j = 0; j < myNode.output_info_size(); j++) {
        //   const auto& out_info = myNode.output_info(j);
        //   if (out_info.shape().dim_size() == 0)
        //     LOG(INFO) << "[Yitao]    child node with size = " << out_info.size() << ", alias_input_port = " << out_info.alias_input_port() << ", shape = " << out_info.shape().dim_size() << " ()" << ", dtype = " << out_info.dtype();
        //   else if (out_info.shape().dim_size() == 1)
        //     LOG(INFO) << "[Yitao]    child node with size = " << out_info.size() << ", alias_input_port = " << out_info.alias_input_port() << ", shape = " << out_info.shape().dim_size() << " (" << out_info.shape().dim(0).size() << ")" << ", dtype = " << out_info.dtype();
        //   else
        //     LOG(INFO) << "[Yitao]    child node with size = " << out_info.size() << ", alias_input_port = " << out_info.alias_input_port() << ", shape = " << out_info.shape().dim_size() << " (" << out_info.shape().dim(0).size() << ", " << out_info.shape().dim(1).size() << ")" << ", dtype = " << out_info.dtype();
        // }
      }
    }
    
    // char data[100];
    // int cost_value;
    // ifstream infile; 
    // infile.open("/home/yitao/Downloads/test/20180910/cost_model_075.txt"); 
    // for (int i = 0; i < 1799; i++) {
    //   infile >> cost_value >> data;
    //   TLS_cost_model->emplace(data, cost_value);
    // }

    *cost_model_generated = true;

    // for (auto it = TLS_cost_model->begin(); it != TLS_cost_model->end(); ++it) {
    //   LOG(INFO) << "[Yitao] TLS_cost_model: " << it->first << " : " << it->second;
    // }
    // Yitao-TLS-End
  }

  // If requested via RunOptions, output the partition graphs.
  if (run_options.output_partition_graphs()) {
    protobuf::RepeatedPtrField<GraphDef>* parition_graph_defs =
        run_metadata->mutable_partition_graphs();
    for (const PerPartitionExecutorsAndLib& exec_and_lib :
         executors_and_keys->items) {
      GraphDef* partition_graph_def = parition_graph_defs->Add();
      exec_and_lib.graph->ToGraphDef(partition_graph_def);
    }
  }

  // // Yitao-TLS-Begin
  // if (true) { // <====== should_we_push_this_node(node) for node level scheduling here
  //   TLS_cv->notify_all();
  // }
  // // Yitao-TLS-End

  // Yitao-TLS-Begin
  {
    olympia_scheduler->SessRunDeregister(sr_info);
  }
  // Yitao-TLS-End

  // LOG(INFO) << "[Yitao] Finished one DirectSession::Run()!";
  LOG(INFO) << "[Yitao] Finished one DirectSession::Run(" << sr_info.sess_id << ", " << sr_info.run_id << ") with " << *cv_check_count << " cv checking!";

  return Status::OK();
}

Status DirectSession::PRunSetup(const std::vector<string>& input_names,
                                const std::vector<string>& output_names,
                                const std::vector<string>& target_nodes,
                                string* handle) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  {
    mutex_lock l(graph_def_lock_);
    if (!graph_created_) {
      return errors::InvalidArgument(
          "Session was not created with a graph before PRunSetup()!");
    }
  }

  // RunOptions is not available in PRunSetup, so use thread pool 0.
  thread::ThreadPool* pool = thread_pools_[0];

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;
  // TODO(cais): TFDBG support for partial runs.
  DebugOptions debug_options;
  RunStateArgs run_state_args(debug_options);
  run_state_args.is_partial_run = true;
  TF_RETURN_IF_ERROR(GetOrCreateExecutors(pool, input_names, output_names,
                                          target_nodes, &executors_and_keys,
                                          &run_state_args));

  // Create the run state and save it for future PRun calls.
  Executor::Args args;
  args.step_id = step_id_counter_.fetch_add(1);
  RunState* run_state =
      new RunState(input_names, output_names, args.step_id, &devices_);
  run_state->rendez = new IntraProcessRendezvous(device_mgr_.get());
  {
    mutex_lock l(executor_lock_);
    if (!partial_runs_
             .emplace(run_state_args.handle,
                      std::unique_ptr<RunState>(run_state))
             .second) {
      return errors::Internal("The handle '", run_state_args.handle,
                              "' created for this partial run is not unique.");
    }
  }

  // Start parallel Executors.
  const size_t num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state->rendez, [run_state](const Status& ret) {
        if (!ret.ok()) {
          mutex_lock l(run_state->mu_);
          run_state->status.Update(ret);
        }
        run_state->executors_done.Notify();
      });

  args.rendezvous = run_state->rendez;
  args.cancellation_manager = cancellation_manager_;
  args.runner = [this, pool](Executor::Args::Closure c) {
    SchedClosure(pool, std::move(c));
  };
  args.session_state = &session_state_;
  args.tensor_store = &run_state->tensor_store;
  args.step_container = &run_state->step_container;
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, run_state_args.handle);
  }
  args.sync_on_finish = sync_on_finish_;

  if (options_.config.graph_options().build_cost_model()) {
    run_state->collector.reset(new StepStatsCollector(nullptr));
    args.stats_collector = run_state->collector.get();
  }

  for (auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }

  *handle = run_state_args.handle;
  return Status::OK();
}

Status DirectSession::PRun(const string& handle, const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           std::vector<Tensor>* outputs) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  std::vector<string> parts = str_util::Split(handle, ';');
  const string& key = parts[0];
  // Get the executors for this partial run.
  ExecutorsAndKeys* executors_and_keys;
  RunState* run_state;
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto exc_it = executors_.find(key);
    if (exc_it == executors_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    executors_and_keys = exc_it->second.get();

    auto prun_it = partial_runs_.find(handle);
    if (prun_it == partial_runs_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    run_state = prun_it->second.get();

    // Make sure that this is a new set of feeds that are still pending.
    for (const auto& input : inputs) {
      auto it = run_state->pending_inputs.find(input.first);
      if (it == run_state->pending_inputs.end()) {
        return errors::InvalidArgument(
            "The feed ", input.first,
            " was not specified in partial_run_setup.");
      } else if (it->second) {
        return errors::InvalidArgument("The feed ", input.first,
                                       " has already been fed.");
      }
    }
    // Check that this is a new set of fetches that are still pending.
    for (const auto& output : output_names) {
      auto it = run_state->pending_outputs.find(output);
      if (it == run_state->pending_outputs.end()) {
        return errors::InvalidArgument(
            "The fetch ", output, " was not specified in partial_run_setup.");
      } else if (it->second) {
        return errors::InvalidArgument("The fetch ", output,
                                       " has already been fetched.");
      }
    }
  }

  // Check that this new set of fetches can be computed from all the
  // feeds we have supplied.
  TF_RETURN_IF_ERROR(
      CheckFetch(inputs, output_names, executors_and_keys, run_state));

  // Send inputs.
  Status s = SendPRunInputs(inputs, executors_and_keys, run_state->rendez);

  // Receive outputs.
  if (s.ok()) {
    s = RecvPRunOutputs(output_names, executors_and_keys, run_state, outputs);
  }

  // Save the output tensors of this run we choose to keep.
  if (s.ok()) {
    s = run_state->tensor_store.SaveTensors(output_names, &session_state_);
  }

  {
    mutex_lock l(executor_lock_);
    // Delete the run state if there is an error or all fetches are done.
    bool done = true;
    if (s.ok()) {
      {
        mutex_lock l(run_state->mu_);
        if (!run_state->status.ok()) {
          LOG(WARNING) << "An error unrelated to this prun has been detected. "
                       << run_state->status;
        }
      }
      for (const auto& input : inputs) {
        auto it = run_state->pending_inputs.find(input.first);
        it->second = true;
      }
      for (const auto& name : output_names) {
        auto it = run_state->pending_outputs.find(name);
        it->second = true;
      }
      done = run_state->PendingDone();
    }
    if (done) {
      WaitForNotification(run_state, cancellation_manager_,
                          operation_timeout_in_ms_);
      partial_runs_.erase(handle);
    }
  }

  return s;
}

Status DirectSession::ResourceHandleToInputTensor(const Tensor& resource_tensor,
                                                  Tensor* retrieved_tensor) {
  if (resource_tensor.dtype() != DT_RESOURCE) {
    return errors::InvalidArgument(strings::StrCat(
        "ResourceHandleToInputTensor() received non-DT_RESOURCE Tensor: ",
        resource_tensor.dtype()));
  }

  ResourceHandle resource_handle = resource_tensor.scalar<ResourceHandle>()();

  if (resource_handle.container() ==
      SessionState::kTensorHandleResourceTypeName) {
    return session_state_.GetTensor(resource_handle.name(), retrieved_tensor);
  } else {
    return errors::InvalidArgument(strings::StrCat(
        "Invalid resource type hash code: ", resource_handle.hash_code(),
        "(name: ", resource_handle.name(),
        " type: ", resource_handle.maybe_type_name(), ")"));
  }
}

Status DirectSession::SendPRunInputs(const NamedTensorList& inputs,
                                     const ExecutorsAndKeys* executors_and_keys,
                                     IntraProcessRendezvous* rendez) {
  Status s;
  Rendezvous::ParsedKey parsed;
  // Insert the input tensors into the local rendezvous by their
  // rendezvous key.
  for (const auto& input : inputs) {
    auto it =
        executors_and_keys->input_name_to_rendezvous_key.find(input.first);
    if (it == executors_and_keys->input_name_to_rendezvous_key.end()) {
      return errors::Internal("'", input.first, "' is not a pre-defined feed.");
    }
    const string& input_key = it->second;

    s = Rendezvous::ParseKey(input_key, &parsed);
    if (!s.ok()) {
      rendez->StartAbort(s);
      return s;
    }

    if (input.second.dtype() == DT_RESOURCE) {
      Tensor tensor_from_handle;
      s = ResourceHandleToInputTensor(input.second, &tensor_from_handle);
      if (s.ok()) {
        s = rendez->Send(parsed, Rendezvous::Args(), tensor_from_handle, false);
      }
    } else {
      s = rendez->Send(parsed, Rendezvous::Args(), input.second, false);
    }

    if (!s.ok()) {
      rendez->StartAbort(s);
      return s;
    }
  }
  return Status::OK();
}

Status DirectSession::RecvPRunOutputs(
    const std::vector<string>& output_names,
    const ExecutorsAndKeys* executors_and_keys, RunState* run_state,
    std::vector<Tensor>* outputs) {
  Status s;
  if (!output_names.empty()) {
    outputs->resize(output_names.size());
  }

  Rendezvous::ParsedKey parsed;
  // Get the outputs from the rendezvous
  for (size_t output_offset = 0; output_offset < output_names.size();
       ++output_offset) {
    const string& output_name = output_names[output_offset];
    auto it =
        executors_and_keys->output_name_to_rendezvous_key.find(output_name);
    if (it == executors_and_keys->output_name_to_rendezvous_key.end()) {
      return errors::Internal("'", output_name,
                              "' is not a pre-defined fetch.");
    }
    const string& output_key = it->second;
    Tensor output_tensor;
    bool is_dead;
    IntraProcessRendezvous* rendez = run_state->rendez;

    s = Rendezvous::ParseKey(output_key, &parsed);
    if (s.ok()) {
      // Fetch data from the Rendezvous.
      s = rendez->Recv(parsed, Rendezvous::Args(), &output_tensor, &is_dead,
                       operation_timeout_in_ms_);
      if (is_dead && s.ok()) {
        s = errors::InvalidArgument("The tensor returned for ", output_name,
                                    " was not valid.");
      }
    }
    if (!s.ok()) {
      rendez->StartAbort(s);
      outputs->clear();
      return s;
    }

    (*outputs)[output_offset] = output_tensor;
  }
  return Status::OK();
}

Status DirectSession::CheckFetch(const NamedTensorList& feeds,
                                 const std::vector<string>& fetches,
                                 const ExecutorsAndKeys* executors_and_keys,
                                 const RunState* run_state) {
  const Graph* graph = executors_and_keys->graph.get();
  const NameNodeMap* name_to_node = &executors_and_keys->name_to_node;

  // Build the set of pending feeds that we haven't seen.
  std::unordered_set<TensorId, TensorId::Hasher> pending_feeds;
  {
    mutex_lock l(executor_lock_);
    for (const auto& input : run_state->pending_inputs) {
      // Skip if the feed has already been fed.
      if (input.second) continue;
      TensorId id(ParseTensorName(input.first));
      auto it = name_to_node->find(id.first);
      if (it == name_to_node->end()) {
        return errors::NotFound("Feed ", input.first, ": not found");
      }
      pending_feeds.insert(id);
    }
  }
  for (const auto& it : feeds) {
    TensorId id(ParseTensorName(it.first));
    pending_feeds.erase(id);
  }

  // Initialize the stack with the fetch nodes.
  std::vector<const Node*> stack;
  for (const string& fetch : fetches) {
    TensorId id(ParseTensorName(fetch));
    auto it = name_to_node->find(id.first);
    if (it == name_to_node->end()) {
      return errors::NotFound("Fetch ", fetch, ": not found");
    }
    stack.push_back(it->second);
  }

  // Any tensor needed for fetches can't be in pending_feeds.
  std::vector<bool> visited(graph->num_node_ids(), false);
  while (!stack.empty()) {
    const Node* n = stack.back();
    stack.pop_back();

    for (const Edge* in_edge : n->in_edges()) {
      const Node* in_node = in_edge->src();
      if (pending_feeds.count({in_node->name(), in_edge->src_output()}) > 0) {
        return errors::InvalidArgument("Fetch ", in_node->name(), ":",
                                       in_edge->src_output(),
                                       " can't be computed from the feeds"
                                       " that have been fed so far.");
      }
      if (!visited[in_node->id()]) {
        visited[in_node->id()] = true;
        stack.push_back(in_node);
      }
    }
  }
  return Status::OK();
}

Status DirectSession::GetOrCreateExecutors(
    thread::ThreadPool* pool, gtl::ArraySlice<string> inputs,
    gtl::ArraySlice<string> outputs, gtl::ArraySlice<string> target_nodes,
    ExecutorsAndKeys** executors_and_keys, RunStateArgs* run_state_args) {
  int64 handle_name_counter_value = -1;
  if (LogMemory::IsEnabled() || run_state_args->is_partial_run) {
    handle_name_counter_value = handle_name_counter_.fetch_add(1);
  }

  string debug_tensor_watches_summary;
  if (!run_state_args->debug_options.debug_tensor_watch_opts().empty()) {
    debug_tensor_watches_summary = SummarizeDebugTensorWatches(
        run_state_args->debug_options.debug_tensor_watch_opts());
  }

  // Fast lookup path, no sorting.
  const string key = strings::StrCat(
      str_util::Join(inputs, ","), "->", str_util::Join(outputs, ","), "/",
      str_util::Join(target_nodes, ","), "/", run_state_args->is_partial_run,
      "/", debug_tensor_watches_summary);
  // Set the handle, if it's needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(key, ";", handle_name_counter_value);
  }

  LOG(INFO) << "[Yitao] @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ key = " << key;

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto it = executors_.find(key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second.get();
      LOG(INFO) << "[Yitao] @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 0...";
      return Status::OK();
    }
  }

  // Slow lookup path, the unsorted key missed the cache.
  // Sort the inputs and outputs, and look up with the sorted key in case an
  // earlier call used a different order of inputs and outputs.
  //
  // We could consider some other signature instead of sorting that
  // preserves the same property to avoid the sort in the future.
  std::vector<string> inputs_sorted(inputs.begin(), inputs.end());
  std::sort(inputs_sorted.begin(), inputs_sorted.end());
  std::vector<string> outputs_sorted(outputs.begin(), outputs.end());
  std::sort(outputs_sorted.begin(), outputs_sorted.end());
  std::vector<string> tn_sorted(target_nodes.begin(), target_nodes.end());
  std::sort(tn_sorted.begin(), tn_sorted.end());

  const string sorted_key = strings::StrCat(
      str_util::Join(inputs_sorted, ","), "->",
      str_util::Join(outputs_sorted, ","), "/", str_util::Join(tn_sorted, ","),
      "/", run_state_args->is_partial_run, "/", debug_tensor_watches_summary);
  // Set the handle, if its needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(sorted_key, ";", handle_name_counter_value);
  }

  LOG(INFO) << "[Yitao] @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ sorted_key = " << sorted_key;

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);
    auto it = executors_.find(sorted_key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second.get();
      // Insert this under the original key.
      executors_.emplace(key, it->second);
      LOG(INFO) << "[Yitao] @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 1...";
      return Status::OK();
    }
  }

  LOG(INFO) << "[Yitao] @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 2...";

  // Nothing found, so create the executors and store in the cache.
  BuildGraphOptions options;
  options.feed_endpoints = inputs_sorted;
  options.fetch_endpoints = outputs_sorted;
  options.target_nodes = tn_sorted;
  options.use_function_convention = !run_state_args->is_partial_run;
  if (!run_state_args->debug_options.debug_tensor_watch_opts().empty()) {
    options.debug_options = run_state_args->debug_options;
  }

  std::shared_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);

  // The executor_lock_ is intentionally released while executor is
  // being created.
  std::unordered_map<string, std::unique_ptr<Graph>> graphs;
  TF_RETURN_IF_ERROR(CreateGraphs(options, &graphs, &ek->flib_def,
                                  run_state_args, &ek->input_types,
                                  &ek->output_types));

  // LOG(INFO) << "[Yitao] @@@@@@ graphs.size() = " << graphs.size() << " @@@@@@";

  if (run_state_args->is_partial_run) {
    ek->graph = std::move(run_state_args->graph);
    std::unordered_set<StringPiece, StringPiece::Hasher> names;
    for (const string& input : inputs) {
      TensorId id(ParseTensorName(input));
      names.emplace(id.first);
    }
    for (const string& output : outputs) {
      TensorId id(ParseTensorName(output));
      names.emplace(id.first);
    }
    for (Node* n : ek->graph->nodes()) {
      if (names.count(n->name()) > 0) {
        ek->name_to_node.insert({n->name(), n});
      }
    }
  }
  ek->items.reserve(graphs.size());
  const auto& optimizer_opts =
      options_.config.graph_options().optimizer_options();
  GraphOptimizer optimizer(optimizer_opts);
  for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
    const string& partition_name = iter->first;
    std::unique_ptr<Graph>& partition_graph = iter->second;

    LOG(INFO) << "[Yitao] @@@@@@ graph.num_nodes() = " << partition_graph->num_nodes() << " @@@@@@ with partition_name = " << partition_name;

    const int graph_def_version = partition_graph->versions().producer();

    Device* device;
    TF_RETURN_IF_ERROR(device_mgr_->LookupDevice(partition_name, &device));

    ek->items.resize(ek->items.size() + 1);
    auto* item = &(ek->items.back());
    item->flib.reset(NewFunctionLibraryRuntime(
        device_mgr_.get(), options_.env, device, graph_def_version,
        ek->flib_def.get(), optimizer_opts));

    LocalExecutorParams params;
    params.device = device;
    params.function_library = item->flib.get();
    auto lib = item->flib.get();
    auto opseg = device->op_segment();
    params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
                                              OpKernel** kernel) {
      // Caches the kernel only if the node is stateful.
      if (!lib->IsStateful(ndef.op())) {
        return lib->CreateKernel(ndef, kernel);
      }
      auto create_fn = [lib, &ndef](OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
      };
      // Kernels created for subgraph nodes need to be cached.  On
      // cache miss, create_fn() is invoked to create a kernel based
      // on the function library here + global op registry.
      return opseg->FindOrCreate(session_handle_, ndef.name(), kernel,
                                 create_fn);
    };
    params.delete_kernel = [lib](OpKernel* kernel) {
      // If the node is stateful, opseg owns it. Otherwise, delete it.
      if (kernel && !lib->IsStateful(kernel->type_string())) {
        delete kernel;
      }
    };
    params.node_outputs_cb = node_outputs_callback_;

    // // Yitao-TLS-Begin
    // // this Optimize(...) is conflict with my node-level scheduling...
    // optimizer.Optimize(lib, options_.env, device, &iter->second);
    // // Yitao-TLS-End

    // EXPERIMENTAL: tfdbg inserts debug nodes in the graph.
    if (!options.debug_options.debug_tensor_watch_opts().empty()) {
      TF_RETURN_IF_ERROR(DecorateAndPublishGraphForDebug(
          options.debug_options, partition_graph.get(), params.device));
    }

    TF_RETURN_IF_ERROR(EnsureMemoryTypes(DeviceType(device->device_type()),
                                         device->name(),
                                         partition_graph.get()));
    // NewLocalExecutor takes ownership of partition_graph.
    item->graph = partition_graph.get();
    item->executor = nullptr;
    Executor* executor;
    TF_RETURN_IF_ERROR(
        NewLocalExecutor(params, partition_graph.release(), &executor));
    item->executor.reset(executor);
  }

  // Cache the mapping from input/output names to graph elements to
  // avoid recomputing it every time.
  if (!run_state_args->is_partial_run) {
    // For regular `Run()`, we use the function calling convention, and so
    // maintain a mapping from input/output names to
    // argument/return-value ordinal index.
    for (size_t i = 0; i < inputs_sorted.size(); ++i) {
      const string& input = inputs_sorted[i];
      ek->input_name_to_index[input] = i;
    }
    for (size_t i = 0; i < outputs_sorted.size(); ++i) {
      const string& output = outputs_sorted[i];
      ek->output_name_to_index[output] = i;
    }
  } else {
    // For `PRun()`, we use the rendezvous calling convention, and so
    // maintain a mapping from input/output names to rendezvous keys.
    //
    // We always use the first device as the device name portion of the
    // key, even if we're feeding another graph.
    for (size_t i = 0; i < inputs_sorted.size(); ++i) {
      const string& input = inputs_sorted[i];
      ek->input_name_to_rendezvous_key[input] = GetRendezvousKey(
          input, device_set_.client_device()->attributes(), FrameAndIter(0, 0));
    }
    for (size_t i = 0; i < outputs_sorted.size(); ++i) {
      const string& output = outputs_sorted[i];
      ek->output_name_to_rendezvous_key[output] =
          GetRendezvousKey(output, device_set_.client_device()->attributes(),
                           FrameAndIter(0, 0));
    }
  }

  // Reacquire the lock, try to insert into the map.
  mutex_lock l(executor_lock_);

  // Another thread may have created the entry before us, in which case we will
  // reuse the already created one.
  auto insert_result = executors_.emplace(sorted_key, ek);
  // Insert the value under the original key, so the fast path lookup will work
  // if the user uses the same order of inputs, outputs, and targets again.
  executors_.emplace(key, insert_result.first->second);
  *executors_and_keys = insert_result.first->second.get();

  return Status::OK();
}

Status DirectSession::CreateGraphs(
    const BuildGraphOptions& subgraph_options,
    std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
    std::unique_ptr<FunctionLibraryDefinition>* flib_def,
    RunStateArgs* run_state_args, DataTypeVector* input_types,
    DataTypeVector* output_types) {

  LOG(INFO) << "[Yitao] @@@@@@ CreateGraphs() is called @@@@@@";

  mutex_lock l(graph_def_lock_);
  std::unique_ptr<SimpleClientGraph> client_graph;

  std::unique_ptr<SimpleGraphExecutionState> temp_exec_state_holder;
  SimpleGraphExecutionState* execution_state = nullptr;
  if (options_.config.graph_options().place_pruned_graph()) {
    // Because we are placing pruned graphs, we need to create a
    // new SimpleGraphExecutionState for every new unseen graph,
    // and then place it.
    SimpleGraphExecutionStateOptions prune_options;
    prune_options.device_set = &device_set_;
    prune_options.session_options = &options_;
    prune_options.stateful_placements = stateful_placements_;
    TF_RETURN_IF_ERROR(SimpleGraphExecutionState::MakeForPrunedGraph(
        execution_state_->original_graph_def().library(), prune_options,
        execution_state_->original_graph_def(), subgraph_options,
        &temp_exec_state_holder, &client_graph));
    execution_state = temp_exec_state_holder.get();
  } else {
    execution_state = execution_state_.get();
    TF_RETURN_IF_ERROR(
        execution_state->BuildGraph(subgraph_options, &client_graph));
  }

  if (subgraph_options.feed_endpoints.size() !=
      client_graph->feed_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of feed endpoints = ",
        subgraph_options.feed_endpoints.size(),
        " versus number of pruned feed endpoints = ",
        client_graph->feed_types.size());
  }
  if (subgraph_options.fetch_endpoints.size() !=
      client_graph->fetch_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of fetch endpoints = ",
        subgraph_options.fetch_endpoints.size(),
        " versus number of pruned fetch endpoints = ",
        client_graph->fetch_types.size());
  }

  auto current_stateful_placements = execution_state->GetStatefulPlacements();
  // Update our current state based on the execution_state's
  // placements.  If there are any mismatches for a node,
  // we should fail, as this should never happen.
  for (auto placement_pair : current_stateful_placements) {
    const string& node_name = placement_pair.first;
    const string& placement = placement_pair.second;
    auto iter = stateful_placements_.find(node_name);
    if (iter == stateful_placements_.end()) {
      stateful_placements_.insert(std::make_pair(node_name, placement));
    } else if (iter->second != placement) {
      return errors::Internal(
          "Stateful placement mismatch. "
          "Current assignment of ",
          node_name, " to ", iter->second, " does not match ", placement);
    }
  }

  stateful_placements_ = execution_state->GetStatefulPlacements();

  // Remember the graph in run state if this is a partial run.
  if (run_state_args->is_partial_run) {
    run_state_args->graph.reset(new Graph(flib_def_.get()));
    CopyGraph(*execution_state->full_graph(), run_state_args->graph.get());
  }

  // Partition the graph across devices.
  PartitionOptions popts;
  popts.node_to_loc = [](const Node* node) {
    return node->assigned_device_name();
  };
  popts.new_name = [this](const string& prefix) {
    return strings::StrCat(prefix, "/_", edge_name_counter_.fetch_add(1));
  };
  popts.get_incarnation = [](const string& name) {
    // The direct session does not have changing incarnation numbers.
    // Just return '1'.
    return 1;
  };
  popts.control_flow_added = false;

  std::unordered_map<string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(Partition(popts, &client_graph->graph, &partitions));

  std::vector<string> device_names;
  for (auto device : devices_) {
    // Extract the LocalName from the device.
    device_names.push_back(DeviceNameUtils::LocalName(device->name()));
  }

  // Check for valid partitions.
  for (const auto& partition : partitions) {
    const string& local_partition_name =
        DeviceNameUtils::LocalName(partition.first);
    if (std::count(device_names.begin(), device_names.end(),
                   local_partition_name) == 0) {
      return errors::InvalidArgument(
          "Creating a partition for ", local_partition_name,
          " which doesn't exist in the list of available devices. Available "
          "devices: ",
          str_util::Join(device_names, ","));
    }
  }

  for (const auto& partition : partitions) {
    std::unique_ptr<Graph> device_graph(
        new Graph(client_graph->flib_def.get()));
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(device_opts, partition.second,
                                              device_graph.get()));
    outputs->emplace(partition.first, std::move(device_graph));
  }

  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = &options_;
  optimization_options.flib_def = client_graph->flib_def.get();
  optimization_options.partition_graphs = outputs;
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PARTITIONING, optimization_options));

  Status s;
  for (auto& partition : *outputs) {
    const string& partition_name = partition.first;
    std::unique_ptr<Graph>* graph = &partition.second;

    VLOG(2) << "Created " << DebugString(graph->get()) << " for "
            << partition_name;

    // Give the device an opportunity to rewrite its subgraph.
    Device* d;
    s = device_mgr_->LookupDevice(partition_name, &d);
    if (!s.ok()) break;
    // TODO(pbar) The library is currently shared and immutable. There
    // may be possible use cases where a device may want to modify
    // function definitions - in which case the library would need to be
    // replicated per device.
    s = d->MaybeRewriteGraph(client_graph->flib_def->ToProto(), graph);
    if (!s.ok()) {
      break;
    }
  }
  *flib_def = std::move(client_graph->flib_def);
  std::swap(*input_types, client_graph->feed_types);
  std::swap(*output_types, client_graph->fetch_types);
  return s;
}

::tensorflow::Status DirectSession::ListDevices(
    std::vector<DeviceAttributes>* response) {
  response->clear();
  response->reserve(devices_.size());
  for (Device* d : devices_) {
    const DeviceAttributes& attrs = d->attributes();
    response->emplace_back(attrs);
  }
  return ::tensorflow::Status::OK();
}

::tensorflow::Status DirectSession::Reset(
    const std::vector<string>& containers) {
  device_mgr_->ClearContainers(containers);
  return ::tensorflow::Status::OK();
}

::tensorflow::Status DirectSession::Close() {
  cancellation_manager_->StartCancel();
  {
    mutex_lock l(closed_lock_);
    if (closed_) return ::tensorflow::Status::OK();
    closed_ = true;
  }
  if (factory_ != nullptr) factory_->Deregister(this);
  return ::tensorflow::Status::OK();
}

DirectSession::RunState::RunState(
    const std::vector<string>& pending_input_names,
    const std::vector<string>& pending_output_names, int64 step_id,
    const std::vector<Device*>* devices)
    : step_container(step_id, [devices](const string& name) {
        for (auto d : *devices) {
          if (!d->resource_manager()->Cleanup(name).ok()) {
            // Do nothing...
          }
        }
      }) {
  // Initially all the feeds and fetches are pending.
  for (auto& name : pending_input_names) {
    pending_inputs[name] = false;
  }
  for (auto& name : pending_output_names) {
    pending_outputs[name] = false;
  }
}

DirectSession::RunState::RunState(int64 step_id,
                                  const std::vector<Device*>* devices)
    : RunState({}, {}, step_id, devices) {}

DirectSession::RunState::~RunState() {
  if (rendez != nullptr) {
    if (!executors_done.HasBeenNotified()) {
      rendez->StartAbort(errors::Cancelled("PRun cancellation"));
      executors_done.WaitForNotification();
    }
    rendez->Unref();
  }
}

bool DirectSession::RunState::PendingDone() const {
  for (const auto& it : pending_inputs) {
    if (!it.second) return false;
  }
  for (const auto& it : pending_outputs) {
    if (!it.second) return false;
  }
  return true;
}

void DirectSession::WaitForNotification(RunState* run_state,
                                        CancellationManager* cm,
                                        int64 timeout_in_ms) {
  Status status =
      WaitForNotification(&run_state->executors_done, timeout_in_ms);
  if (!status.ok()) {
    {
      mutex_lock l(run_state->mu_);
      run_state->status.Update(status);
    }
    cm->StartCancel();
    // We must wait for the executors to complete, because they have borrowed
    // references to `cm` and other per-step state. After this notification, it
    // is safe to clean up the step.
    run_state->executors_done.WaitForNotification();
  }
}

::tensorflow::Status DirectSession::WaitForNotification(
    Notification* notification, int64 timeout_in_ms) {
  if (timeout_in_ms > 0) {
    int64 timeout_in_us = timeout_in_ms * 1000;
    bool notified = WaitForNotificationWithTimeout(notification, timeout_in_us);
    if (!notified) {
      return Status(error::DEADLINE_EXCEEDED,
                    "Timed out waiting for notification");
    }
  } else {
    notification->WaitForNotification();
  }
  return Status::OK();
}

}  // namespace tensorflow
