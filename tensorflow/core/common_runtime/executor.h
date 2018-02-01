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

#ifndef TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_
#define TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

// Yitao-TLS-Begin
#include <thread>
#include <queue>
#include <list>
// a simple class to store the Session Id and Sess.run() Id
class SessRunInfo {
public:
  SessRunInfo() {
    sess_id = -1;
    run_id = -1;
  }
  SessRunInfo(int tt_sess_id, int tt_run_id) {
    sess_id = tt_sess_id;
    run_id = tt_run_id;
  }

  int sess_id;
  int run_id;

  bool operator==(const SessRunInfo &other) const {
    return (sess_id == other.sess_id && run_id == other.run_id);
  }

  bool operator!=(const SessRunInfo &other) const {
    return (sess_id != other.sess_id || run_id != other.run_id);
  }

  bool operator< (const SessRunInfo &other) const {
    if (sess_id != other.sess_id)
      // sess_id < other.sess_id ===> (1, *) < (2, *)
      return sess_id < other.sess_id;
    else
      // run_id < other.run_id ===> (a, 1) < (a, 2)
      return run_id < other.run_id;
  }

  // friend ostream& operator<< (ostream& os, const SessRunInfo& sr_info);
};

// ostream& operator<< (ostream& os, const SessRunInfo& sr_info) {
//   os << "(" << sr_info.sess_id << ", " << sr_info.run_id << ")";
//   return os;
// }

namespace std {
  template <>
  struct hash<SessRunInfo> {
    size_t operator()(const SessRunInfo& sr_info) const {
      return ((hash<int>()(sr_info.sess_id) ^ (hash<int>()(sr_info.run_id) << 1)) >> 1);
    }
  };
}

template<typename T>
class CustomPriorityQueue : public std::priority_queue<T, std::vector<T>> {
public:
  bool remove(const T& value) {
    auto it = std::find(this->c.begin(), this->c.end(), value);
    if (it != this->c.end()) {
      this->c.erase(it);
      std::make_heap(this->c.begin(), this->c.end(), this->comp);
      return true;
    } else {
      return false;
    }
  }
};

class OlympiaScheduler{
public:
  OlympiaScheduler() {
    // cout << "OlympiaScheduler initialized..." << endl;
    token_info = SessRunInfo(-1, -1);
    // notify_done = new bool;
  }

  void SessRunRegister(SessRunInfo sr_info) {
    std::unique_lock<std::mutex> lk(sched_lock);
    sr_queue.push(sr_info);
    // sr_queue.push_back(sr_info);
    // if (sr_info.run_id >= 15 && (sr_info.run_id - 15) % 10 < 5) {
    //   // weighted fair sharing
    //   for (int i = 0; i < 9; i++)
    //     sr_queue.push_back(sr_info);
    // }
    std::condition_variable* my_cv = new std::condition_variable;
    cv_map[sr_info] = my_cv;
    // cumulate_cost_map[sr_info] = my_cumulated_cost;

    // should update token_info and notify the corresponding cv here!!! <<<<<<<<<<<<<<<<<<<<
    // if (sr_queue.size() == 1) {    // <=============== pay attention! need to deal with only one job running, so we need to update token
    //   token_info = sr_queue.top();
    //   LOG(INFO) << "[Yitao] in SessRunRegister(" << sr_info.sess_id << ", " << sr_info.run_id << "), let's notify (" << token_info.sess_id << ", " << token_info.run_id << ")...";
    //   std::condition_variable* cur_cv = cv_map[token_info];
    //   lk.unlock();
    //   cur_cv->notify_all();
    // } else {
    //   LOG(INFO) << "[Yitao] in SessRunRegister(" << sr_info.sess_id << ", " << sr_info.run_id << "), let's previous token keep running...";
    //   lk.unlock();
    // }

    // if (sr_info.run_id == 15) {
    //   if (sr_queue.size() == 1) {
    //     LOG(INFO) << "[Yitao] in SessRunRegister(*, 15), we only have one model, let's wait...";
    //     lk.unlock();
    //     return;
    //   } else {
    //     LOG(INFO) << "[Yitao] in SessRunRegister(*, 15), now we have two models, let's run!";
    //     token_info = sr_queue.top();
    //     std::condition_variable* cur_cv = cv_map[token_info];
    //     lk.unlock();
    //     cur_cv->notify_all();
    //     return;
    //   }
    // }

    // if (sr_info.run_id == 15 || sr_info.run_id == 16) {
    //   if (sr_queue.size() == 1) {
    //     LOG(INFO) << "[Yitao] in SessRunRegister(" << sr_info.sess_id << ", " << sr_info.run_id << "), we only have one request, let's wait...";
    //     lk.unlock();
    //     return;
    //   } else {
    //     LOG(INFO) << "[Yitao] in SessRunRegister(" << sr_info.sess_id << ", " << sr_info.run_id << "), now we have two requests, let's run!";
    //     // token_info = sr_queue.top();
    //     token_info = sr_queue.front();
    //     std::condition_variable* cur_cv = cv_map[token_info];
    //     lk.unlock();
    //     cur_cv->notify_all();
    //     return;
    //   }
    // }

    // // Fair sharing special sync...
    // int target_parallel_job = 80;
    // if (sr_info.run_id >= 15) {
    //   if (sr_queue.size() != target_parallel_job) {
    //     LOG(INFO) << "[Yitao] in SessRunRegister(" << sr_info.sess_id << ", " << sr_info.run_id << "), we only have " << sr_queue.size() << " request, let's wait...";
    //     lk.unlock();
    //     return;
    //   } else {
    //     LOG(INFO) << "[Yitao] in SessRunRegister(" << sr_info.sess_id << ", " << sr_info.run_id << "), now we have " << target_parallel_job << " requests, let's run!";
    //     token_info = sr_queue.front();
    //     std::condition_variable* cur_cv = cv_map[token_info];
    //     lk.unlock();
    //     cur_cv->notify_all();
    //     return;
    //   }
    // }

    token_info = sr_queue.top();
    // token_info = sr_queue.front();
    LOG(INFO) << "[Yitao] in SessRunRegister(" << sr_info.sess_id << ", " << sr_info.run_id << "), change token to (" << token_info.sess_id << ", " << token_info.run_id << ")...";
    std::condition_variable* cur_cv = cv_map[token_info];
    lk.unlock();
    cur_cv->notify_all();
  }

  void SessRunDeregister(SessRunInfo sr_info) {
    std::unique_lock<std::mutex> lk(sched_lock);
    sr_queue.remove(sr_info);
    
    // for (std::list<SessRunInfo>::iterator it = sr_queue.begin(); it != sr_queue.end(); ) {
    //   if (*it == sr_info) {
    //     it = sr_queue.erase(it);
    //     // break;
    //   } else {
    //     ++it;
    //   }
    // }

    // cv_map.erase(sr_info);
    // cumulate_cost_map.erase(sr_info);

    // should update token_info and notify the corresponding cv here!!! <<<<<<<<<<<<<<<<<<<<
    token_info = sr_queue.top();
    // token_info = sr_queue.front();
    LOG(INFO) << "[Yitao] in SessRunDeregister(" << sr_info.sess_id << ", " << sr_info.run_id << "), let's notify (" << token_info.sess_id << ", " << token_info.run_id << ")...";
    std::condition_variable* cur_cv = cv_map[token_info];
    lk.unlock();
    cur_cv->notify_all();
  }

  void SessRunUpdateTokenInfo(SessRunInfo sr_info, const tensorflow::Node* node, int process_id) {
    // Case Default
    std::unique_lock<std::mutex> lk(sched_lock);

    // should update token_info and notify the corresponding cv here!!! <<<<<<<<<<<<<<<<<<<<
    token_info = sr_queue.top();

    if (token_info != sr_info) {
      LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), let's switch to (" << token_info.sess_id << ", " << token_info.run_id << ")! on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
      std::condition_variable* cur_cv = cv_map[token_info];
      lk.unlock();
      cur_cv->notify_all();
    } else {
      LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), keep the same SessRun... on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
      lk.unlock();
    }

    // // Case Two
    // std::unique_lock<std::mutex> lk(sched_lock);
    // if (sr_queue.size() == 1) {
    //   token_info = sr_info;
    //   LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), keep the same SessRun... on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
    //   lk.unlock();
    // } else if (sr_queue.size() == 2) {
    //   if (sr_info.run_id == 15) {
    //     token_info.run_id = 16;
    //   } else {
    //     token_info.run_id = 15;
    //   }
    //   LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), let's switch to (" << token_info.sess_id << ", " << token_info.run_id << ")! on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
    //   std::condition_variable* cur_cv = cv_map[token_info];
    //   lk.unlock();
    //   cur_cv->notify_all();
    // } else {
    //   LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), @@@@@@ BugBugBug @@@@@@... on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
    //   lk.unlock();
    // }

    // // Case Three
    // std::unique_lock<std::mutex> lk(sched_lock);
    // if (sr_queue.size() == 1) {
    //   token_info = sr_info;
    //   LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), keep the same SessRun... on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
    //   lk.unlock();
    // } else if (sr_queue.size() == 2) {
    //   if (sr_info == SessRunInfo(0, 15)) {
    //     token_info.sess_id = 1;
    //     token_info.run_id = 15;
    //   } else {
    //     token_info.sess_id = 0;
    //     token_info.run_id = 15;
    //   }
    //   LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), let's switch to (" << token_info.sess_id << ", " << token_info.run_id << ")! on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
    //   std::condition_variable* cur_cv = cv_map[token_info];
    //   lk.unlock();
    //   cur_cv->notify_all();
    // } else {
    //   LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), @@@@@@ BugBugBug @@@@@@... on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
    //   lk.unlock();
    // }

    // // Case Four
    // std::unique_lock<std::mutex> lk(sched_lock);
    // if (sr_queue.size() == 1) {
    //   token_info = sr_info;
    //   LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), keep the same SessRun... on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
    //   lk.unlock();
    // } else if (sr_queue.size() == 2) {
    //   if (sr_info.run_id == 15) {
    //     token_info.run_id = 16;
    //   } else {
    //     token_info.run_id = 15;
    //   }
    //   LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), let's switch to (" << token_info.sess_id << ", " << token_info.run_id << ")! on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
    //   std::condition_variable* cur_cv = cv_map[token_info];
    //   lk.unlock();
    //   cur_cv->notify_all();
    // } else {
    //   LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), @@@@@@ BugBugBug @@@@@@... on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
    //   lk.unlock();
    // }

    // // Fair sharing
    // std::unique_lock<std::mutex> lk(sched_lock);
    // if (sr_queue.size() == 1) {
    //   token_info = sr_info;
    //   LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), keep the same SessRun... on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
    //   lk.unlock();
    // } else {
    //   SessRunInfo tmp_info = sr_queue.front();
    //   sr_queue.pop_front();
    //   sr_queue.push_back(tmp_info);
    //   token_info = sr_queue.front();
    //   LOG(INFO) << "[Yitao] in SessRunUpdateTokenInfo(" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << "), let's switch to (" << token_info.sess_id << ", " << token_info.run_id << ")! on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
    //   std::condition_variable* cur_cv = cv_map[token_info];
    //   lk.unlock();
    //   cur_cv->notify_all();
    // }
  }

  void SessRunYieldOrRun(SessRunInfo sr_info, const tensorflow::Node* node, int process_id) {
    // // case 2: no bug, see out27-case2
    // return;
    // // case 5: has bug, see out26-case5
    // if (sr_info.run_id == 19)
    //   return;
    std::unique_lock<std::mutex> lk(sched_lock);
    // // case 6: has bug, see out29-case6
    // if (sr_info == this->token_info)
    //   return;
    std::condition_variable* my_cv = cv_map[sr_info];
    my_cv->wait(lk, [sr_info, node, process_id, this]() {
      // // case 3: no bug see out28-case3
      // return true;
      // // case 4: has bug, see out25-case4
      // if (sr_info.run_id == 19)
      //   return true;
      bool tmp = (sr_info == this->token_info);
      if (!tmp) {
        // LOG(INFO) << "[Yitao] cv.wait: sr_info = (" << sr_info.sess_id << ", " << sr_info.run_id << ") != token_info = (" << token_info.sess_id << ", " << token_info.run_id << "), thread suspended";
        LOG(INFO) << "[Yitao] cv.wait: sr_info = (" << sr_info.sess_id << ", " << sr_info.run_id << ", " << process_id << ") != token_info = (" << token_info.sess_id << ", " << token_info.run_id << "), thread suspended on Node " << node->id() << " " << node->type_string() << " " << node->name() << " on device " << node->assigned_device_name() << " in process " << process_id;
      }
      return tmp;
    });

    lk.unlock();
  }

  // std::mutex* GetSchedLock() {
  //   return sched_lock;
  // }

  // SessRunInfo GetCurSessRunInfo() {
  //   return token_info;
  // }

private:
  SessRunInfo token_info;
  std::unordered_map<SessRunInfo, std::condition_variable*> cv_map;
  // std::unordered_map<SessRunInfo, int*> cumulate_cost_map;
  // bool* notify_done;
  std::mutex sched_lock;

  CustomPriorityQueue<SessRunInfo> sr_queue;
  // std::list<SessRunInfo> sr_queue;
};
// Yitao-TLS-End

namespace tensorflow {



class StepStatsCollector;

// Executor runs a graph computation.
// Example:
//   Graph* graph = ...;
//      ... construct graph ...
//   Executor* executor;
//   TF_CHECK_OK(NewSimpleExecutor(my_device, graph, &executor));
//   Rendezvous* rendezvous = NewNaiveRendezvous();
//   TF_CHECK_OK(rendezvous->Send("input", some_input_tensor));
//   TF_CHECK_OK(executor->Run({ExecutorOpts, rendezvous, nullptr}));
//   TF_CHECK_OK(rendezvous->Recv("output", &output_tensor));
//   ... ...
//
// Multiple threads can call Executor::Run concurrently.
class Executor {
 public:
  virtual ~Executor() {}

  // RunAsync() executes the graph computation. "done" is run when the
  // graph computation completes. If any error happens during the
  // computation, "done" is run and the error is passed to "done".
  //
  // RunAsync() is given a few arguments in Args. The caller must
  // ensure objects passed in Args (rendezvous, stats_collector, etc.)
  // are alive at least until done is invoked. All pointers to the
  // argument objects can be nullptr.
  //
  // "step_id" is a process-wide unique identifier for the step being
  // run. Executors on different devices may receive the same step_id
  // in the case that a step runs Ops on more than one device. The
  // step_id is used for tracking resource usage of a given step.
  //
  // RunAsync() uses the given "rendezvous", if not null, as the
  // mechanism to communicate inputs and outputs of the underlying
  // graph computation.
  //
  // RunAsync() calls "stats_collector", if not null, to keep track of
  // stats. This allows us to collect statistics and traces on demand.
  //
  // RunAsync() is provided a "call_frame", if the executor is used
  // for executing a function, is used to pass arguments and return
  // values between the caller and the callee.
  //
  // RunAsync() uses "cancellation_manager", if not nullptr, to
  // register callbacks that should be called if the graph computation
  // is canceled. Note that the callbacks merely unblock any
  // long-running computation, and a canceled step will terminate by
  // returning/calling the DoneCallback as usual.
  //
  // RunAsync() dispatches closures to "runner". Typically, "runner"
  // is backed up by a bounded threadpool.
  struct Args {
    int64 step_id = 0;
    Rendezvous* rendezvous = nullptr;
    StepStatsCollector* stats_collector = nullptr;
    FunctionCallFrame* call_frame = nullptr;
    CancellationManager* cancellation_manager = nullptr;
    SessionState* session_state = nullptr;
    TensorStore* tensor_store = nullptr;
    ScopedStepContainer* step_container = nullptr;

    // If true, calls Sync() on the device.
    bool sync_on_finish = false;

    typedef std::function<void()> Closure;
    typedef std::function<void(Closure)> Runner;
    Runner runner = nullptr;

    // A callback that is invoked each time a node has finished executing.
    typedef std::function<Status(const string& node_name, const int output_slot,
                                 const Tensor* tensor, const bool is_ref,
                                 OpKernelContext* ctx)>
        NodeOutputsCallback;
    NodeOutputsCallback node_outputs_cb = nullptr;

    // Yitao-TLS-Begin
    // int sess_id;

    // int* next_sess_id;
    // int* next_sess_run_id;

    // bool* notify_done;

    // std::mutex* sched_lock; // shared by both TLS_cv and sched_cv
    // std::condition_variable* TLS_cv;
    // std::condition_variable* sched_cv;

    // std::priority_queue<sessRunInfo>* TLS_queue;

    // int sess_run_id;

    // *** Per Session
    bool* cost_model_generated;
    std::unordered_map<string, int>* TLS_cost_model; 
    OlympiaScheduler* olympia_scheduler;

    // *** Per Session::Run
    int* cv_check_count;
    SessRunInfo sr_info;
    // std::condition_variable* my_cv;
    int* my_cumulated_cost;
    std::mutex* my_lock;
    // Yitao-TLS-End

  };
  typedef std::function<void(const Status&)> DoneCallback;
  virtual void RunAsync(const Args& args, DoneCallback done) = 0;

  // Synchronous wrapper for RunAsync().
  Status Run(const Args& args) {
    Status ret;
    Notification n;
    RunAsync(args, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }
};

// Creates an Executor that computes the given "graph".
//
// If successful, returns the constructed executor in "*executor". The
// caller keeps the ownership of "device". The returned executor takes
// the ownership of "graph". Otherwise, returns an error status.
//
// "params" provides a set of context for the executor. We expect that
// different context would provide different implementations.
struct LocalExecutorParams {
  Device* device;

  // The library runtime support.
  FunctionLibraryRuntime* function_library = nullptr;

  // create_kernel returns an instance of op kernel based on NodeDef.
  // delete_kernel is called for every kernel used by the executor
  // when the executor is deleted.
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
  std::function<void(OpKernel*)> delete_kernel;

  Executor::Args::NodeOutputsCallback node_outputs_cb;
};
::tensorflow::Status NewLocalExecutor(const LocalExecutorParams& params,
                                      const Graph* graph, Executor** executor);

// A class to help run multiple executors in parallel and wait until
// all of them are complete.
//
// ExecutorBarrier deletes itself after the function returned by Get()
// is called.
class ExecutorBarrier {
 public:
  typedef std::function<void(const Status&)> StatusCallback;

  // Create an ExecutorBarrier for 'num' different executors.
  //
  // 'r' is the shared Rendezvous object that is used to communicate
  // state.  If any of the executors experiences an error, the
  // rendezvous object will be aborted exactly once.
  //
  // 'done' is called after the last executor completes, and
  // ExecutorBarrier is deleted.
  ExecutorBarrier(size_t num, Rendezvous* r, StatusCallback done)
      : rendez_(r), done_cb_(done), pending_(num) {}

  ~ExecutorBarrier() {}

  // Returns a closure that Executors must call when they are done
  // computing, passing the status of their execution as an argument.
  StatusCallback Get() {
    return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
  }

 private:
  Rendezvous* rendez_ = nullptr;
  StatusCallback done_cb_ = nullptr;

  mutable mutex mu_;
  int pending_ GUARDED_BY(mu_) = 0;
  Status status_ GUARDED_BY(mu_);

  void WhenDone(const Status& s) {
    bool error = false;
    Rendezvous* error_rendez = nullptr;
    StatusCallback done = nullptr;
    Status status;
    {
      mutex_lock l(mu_);
      // If we are the first error encountered, mark the status
      // appropriately and later trigger an abort of the Rendezvous
      // object by this thread only.
      if (status_.ok() && !s.ok()) {
        error = true;
        error_rendez = rendez_;
        error_rendez->Ref();
        status_ = s;
      }

      // If this is the last call to WhenDone, call the final callback
      // below.
      if (--pending_ == 0) {
        CHECK(done_cb_ != nullptr);
        done = done_cb_;
        done_cb_ = nullptr;
      }

      status = status_;
    }

    if (error) {
      error_rendez->StartAbort(status);
      error_rendez->Unref();
    }
    if (done != nullptr) {
      delete this;
      done(status);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorBarrier);
};

// A few helpers to facilitate create/delete kernels.

// Creates a kernel based on "ndef" on device "device". The kernel can
// access the functions in the "flib". The caller takes ownership of
// returned "*kernel".
Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const NodeDef& ndef, int graph_def_version,
                             OpKernel** kernel);

// Deletes "kernel" returned by CreateKernel.
void DeleteNonCachedKernel(OpKernel* kernel);

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_
