#ifndef __wbCUDA_NVTX_h__
#define __wbCUDA_NVTX_h__

#include "wbConfiguration.h"

#ifdef NVTX_Enable
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"
#include "nvToolsExtCudaRt.h"

// You'll need to add the following to your project:
//  Include Directory:      $(NVTOOLSEXT_PATH)include
//  Library Directory:      $(NVTOOLSEXT_PATH)lib\$(Platform)
// 
// And you'll need to setup your project to copy $(NVTOOLSEXT_PATH)\bin\$(Platform)\nvToolsExt64_*.dll.
// The NVIDIA examples use the following addition to the project file to accomplish this copy:
//
//      <!--Files you need-->
//      <ItemGroup>
//          <FilesToCopy Include = "$(NVTOOLSEXT_PATH)\bin\$(Platform)\nvToolsExt64_*.dll" / >
//      </ItemGroup>
//
// Alternatively, setup a nvToolsExt64_1.ref file in your project and mark it as build with "Custom Build Tool":
//      Command Line:               COPY "$(NVTOOLSEXT_PATH)bin\$(Platform)\nvToolsExt64_1.dll" "$(TargetDir)"
//      Description:                Copying nvToolsExt64_1.dll to target folder...
//      Outputs:                    $(TargetDir)nvToolsExt64_1.dll
//      Additional Dependencies:    $(NVTOOLSEXT_PATH)bin\$(Platform)\nvToolsExt64_1.dll
#pragma comment(lib, "nvToolsExt64_1.lib")
#endif

// ============================================================================
// PREPROCESSOR
// ============================================================================

#ifdef NVTX_Enable
#define DELAY_RANGE() Sleep(100);     // sleep 100 ms
#define DELAY()       Sleep(10);      // sleep 10 ms
#else
#define DELAY_RANGE()
#define DELAY()
#endif

// C Preprocessor macros to conditionally enable NvToolsExt functions.

#if 0       // use nvtx namespace instead.
#ifdef NVTX_Enable

#define NVTX_MarkEx nvtxMarkEx
#define NVTX_MarkA nvtxMarkA
#define NVTX_MarkW nvtxMarkW

#define NVTX_RangeStartEx nvtxRangeStartEx
#define NVTX_RangeStartA nvtxRangeStartA
#define NVTX_RangeStartW nvtxRangeStartW

#define NVTX_RangeEnd nvtxRangeEnd

#define NVTX_RangePushEx nvtxRangePushEx
#define NVTX_RangePushA nvtxRangePushA
#define NVTX_RangePushW nvtxRangePushW

#define NVTX_RangePop nvtxRangePop

#define NVTX_NameOsThreadA nvtxNameOsThreadA
#define NVTX_NameOsThreadW nvtxNameOsThreadW

#else

#define NVTX_MarkEx __noop
#define NVTX_MarkA __noop
#define NVTX_MarkW __noop

#define NVTX_RangeStartEx __noop
#define NVTX_RangeStartA __noop
#define NVTX_RangeStartW __noop

#define NVTX_RangeEnd __noop

#define NVTX_RangePushEx __noop
#define NVTX_RangePushA __noop
#define NVTX_RangePushW __noop

#define NVTX_RangePop __noop

#define NVTX_NameOsThreadA __noop
#define NVTX_NameOsThreadW __noop

#endif
#endif

// C++ function templates to enable NvToolsExt functions
namespace nvtx
{
#ifdef NVTX_Enable
    class Attributes
    {
    public:
        inline Attributes()
        {
            clear();
        }

        inline Attributes& category(uint32_t category)
        {
            m_event.category = category;
            return *this;
        }

        inline Attributes& color(uint32_t argb)
        {
            m_event.colorType = NVTX_COLOR_ARGB;
            m_event.color = argb;
            return *this;
        }

        inline Attributes& payload(uint64_t value)
        {
            m_event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
            m_event.payload.ullValue = value;
            return *this;
        }

        inline Attributes& payload(int64_t value)
        {
            m_event.payloadType = NVTX_PAYLOAD_TYPE_INT64;
            m_event.payload.llValue = value;
            return *this;
        }

        inline Attributes& payload(double value)
        {
            m_event.payloadType = NVTX_PAYLOAD_TYPE_DOUBLE;
            m_event.payload.dValue = value;
            return *this;
        }

        inline Attributes& message(const char* message)
        {
            m_event.messageType = NVTX_MESSAGE_TYPE_ASCII;
            m_event.message.ascii = message;
            return *this;
        }

        inline Attributes& message(const wchar_t* message)
        {
            m_event.messageType = NVTX_MESSAGE_TYPE_UNICODE;
            m_event.message.unicode = message;
            return *this;
        }

        inline Attributes& clear()
        {
            memset(&m_event, 0, NVTX_EVENT_ATTRIB_STRUCT_SIZE);
            m_event.version = NVTX_VERSION;
            m_event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            return *this;
        }

        inline const nvtxEventAttributes_t* out() const
        {
            return &m_event;
        }

    private:
        nvtxEventAttributes_t m_event;
    };


    class ScopedRange
    {
    public:
        inline ScopedRange(const char* message)
        {
            nvtxRangePushA(message);
        }

        inline ScopedRange(const wchar_t* message)
        {
            nvtxRangePushW(message);
        }

        inline ScopedRange(const nvtxEventAttributes_t* attributes)
        {
            nvtxRangePushEx(attributes);
        }

        inline ScopedRange(const nvtx::Attributes& attributes)
        {
            nvtxRangePushEx(attributes.out());
        }

        inline ~ScopedRange()
        {
            nvtxRangePop();
        }
    };

    inline void Mark(const nvtx::Attributes& attrib) { nvtxMarkEx(attrib.out()); }
    inline void Mark(const nvtxEventAttributes_t* eventAttrib) { nvtxMarkEx(eventAttrib); }
    inline void Mark(const char* message) { nvtxMarkA(message); }
    inline void Mark(const wchar_t* message) { nvtxMarkW(message); }

    inline nvtxRangeId_t RangeStart(const nvtx::Attributes& attrib) { return nvtxRangeStartEx(attrib.out()); }
    inline nvtxRangeId_t RangeStart(const nvtxEventAttributes_t* eventAttrib) { return nvtxRangeStartEx(eventAttrib); }
    inline nvtxRangeId_t RangeStart(const char* message) { return nvtxRangeStartA(message); }
    inline nvtxRangeId_t RangeStart(const wchar_t* message) { return nvtxRangeStartW(message); }

    inline void RangeEnd(nvtxRangeId_t id) { nvtxRangeEnd(id); }

    inline int RangePush(const nvtx::Attributes& attrib) { return nvtxRangePushEx(attrib.out()); }
    inline int RangePush(const nvtxEventAttributes_t* eventAttrib) { return nvtxRangePushEx(eventAttrib); }
    inline int RangePush(const char* message) { return nvtxRangePushA(message); }
    inline int RangePush(const wchar_t* message) { return nvtxRangePushW(message); }

    inline void RangePop() { nvtxRangePop(); }

    inline void SetNameCategory(uint32_t category, const char* name) { nvtxNameCategoryA(category, name); }
    inline void SetNameCategory(uint32_t category, const wchar_t* name) { nvtxNameCategoryW(category, name); }

    inline void SetNameOsThread(uint32_t threadId, const char* name) { nvtxNameOsThreadA(threadId, name); }
    inline void SetNameOsThread(uint32_t threadId, const wchar_t* name) { nvtxNameOsThreadW(threadId, name); }
    inline void SetNameCurrentThread(const char* name) { nvtxNameOsThreadA(::GetCurrentThreadId(), name); }
    inline void SetNameCurrentThread(const wchar_t* name) { nvtxNameOsThreadW(::GetCurrentThreadId(), name); }

    inline void SetName(cudaStream_t cudaStream, const char* name) { nvtxNameCudaStreamA(cudaStream, name); }
    inline void SetName(CUcontext cuContext, const char* name) { nvtxNameCuContextA(cuContext, name); }

#else

    class Attributes
    {
    public:
        Attributes() {}
        Attributes& category(uint32_t category) { return *this; }
        Attributes& color(uint32_t argb) { return *this; }
        Attributes& payload(uint64_t value) { return *this; }
        Attributes& payload(int64_t value) { return *this; }
        Attributes& payload(double value) { return *this; }
        Attributes& message(const char* message) { return *this; }
        Attributes& message(const wchar_t* message) { return *this; }
        Attributes& clear() { return *this; }
        const nvtxEventAttributes_t* out() { return 0; }
    };

    class ScopedRange
    {
    public:
        ScopedRange(const char* message) { (void)message; }
        ScopedRange(const wchar_t* message) { (void)message; }
        ScopedRange(const nvtxEventAttributes_t* attributes) { (void)attributes; }
        ScopedRange(const Attributes& attributes) { (void)attributes; }
        ~ScopedRange() {}
    };

    inline void Mark(const nvtx::Attributes& attrib) { (void)attrib; }
    inline void Mark(const nvtxEventAttributes_t* eventAttrib) { (void)eventAttrib; }
    inline void Mark(const char* message) { (void)message; }
    inline void Mark(const wchar_t* message) { (void)message; }

    inline nvtxRangeId_t RangeStart(const nvtx::Attributes& attrib) { (void)attrib; return 0; }
    inline nvtxRangeId_t RangeStart(const nvtxEventAttributes_t* eventAttrib) { (void)eventAttrib; return 0; }
    inline nvtxRangeId_t RangeStart(const char* message) { (void)message; return 0; }
    inline nvtxRangeId_t RangeStart(const wchar_t* message) { (void)message; return 0; }

    inline void RangeEnd(nvtxRangeId_t id) { (void)id; }


    inline int RangePush(const nvtx::Attributes& attrib) { (void)attrib; return -1; }
    inline int RangePush(const nvtxEventAttributes_t* eventAttrib) { (void)eventAttrib; return -1; }
    inline int RangePush(const char* message) { (void)message; return -1; }
    inline int RangePush(const wchar_t* message) { (void)message; return -1; }

    inline int RangePop() { return -1; }

    inline void SetNameCategory(uint32_t category, const char* name) { (void)category; (void)name; }
    inline void SetNameCategory(uint32_t category, const wchar_t* name) { (void)category; (void)name; }

    inline void SetNameOsThread(uint32_t threadId, const char* name) { (void)threadId; (void)name; }
    inline void SetNameOsThread(uint32_t threadId, const wchar_t* name) { (void)threadId; (void)name; }
    inline void SetNameCurrentThread(const char* name) { (void)name; }
    inline void SetNameCurrentThread(const wchar_t* name) { (void)name; }

    inline void SetName(cudaStream_t cudaStream, const char* name) { (void)cudaStream; (void)name; }
    inline void SetName(CUcontext cuContext, const char* name) { (void)cuContext; (void)name; }

#endif
}

#endif	// __wbCUDA_NVTX_h__

