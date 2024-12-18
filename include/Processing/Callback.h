/*	Callback.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)

	Sets up function/member callback in a relatively clean fashion.  Supports up to 4 arguments (more could be allowed) but the return
	value must be void in this implementation.

	Usage example #1 -------------------------------

	class Example
	{
	public:
		void MyFunc(int i) { printf("Event %i happened!\n", i); }
	};

	void main()
	{
		Example myobj;

		typedef Callback<int> MyCallback;
		MyCallback callback = new MyCallback::Member<Example>(&myobj, &Example::MyFunc);
		callback(5);
	}

	Usage example #2 -------------------------------	

	class Service
	{
	public:
		struct Data { ... };

		typedef Callback<Data&> DataHandler;
		void Generate(DataHandler OnData)
		{
			// generate SomeData...
			OnData(SomeData);
		}
	};

	class Caller
	{
	public:
		void UseTheData(Service::Data& Content) { ... }
	};

	void main()
	{
		Caller Joe;
		Service Voicemail;

		Voicemail.Generate(new Service::DataHandler::Member<Caller>(&Joe, &Caller::UseTheData));
	}

	----------------------------------------------

	For a static or non-member function, use ::Static instead of ::Member.
*/

#ifndef __WBCallback_h__
#define __WBCallback_h__

#include "../wbFoundation.h"

namespace wb
{
	/** This solution is derived from the CodeProject article:
		http://www.codeproject.com/Articles/6197/Emulating-C-delegates-in-Standard-C
		by Arnold the Aardvark.  Good stuff. **/

	/** It's a shame that variadic templates aren't well supported at this time, but I've found a workaround that makes for a lengthy
		but workable solution through partial specialization and default template parameters. **/

	/** Boost::bind and Boost::function probably do this better, but for now... **/

	template <class Arg1 = void, class Arg2 = void, class Arg3 = void, class Arg4 = void>
	class Callback
	{
	private:	

		class Base
		{
		public:
			virtual void operator()(Arg1, Arg2, Arg3, Arg4) = 0;
			virtual Base* Clone() = 0;
			virtual ~Base() { }
		};

		Base*	mPtr;
    
	public:  

		/** CallbackX<...>::Member should be used when setting up a callback to a member function **/
		template <typename Class>
		class Member : public Base
		{		
			typedef void (Class::*FuncPtr)(Arg1, Arg2, Arg3, Arg4);

			Class*		mObjectPtr;		// Pointer to the object we are delegating to.
			FuncPtr		mFuncPtr;		// Address of the function on the delegate object.

		public:
			Member(Class* ObjectPtr, FuncPtr FuncPtr) : mObjectPtr(ObjectPtr), mFuncPtr(FuncPtr) { }
     
			void operator()(Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4) override
			{
				(mObjectPtr->*mFuncPtr)(arg1, arg2, arg3, arg4);
			}

			Base* Clone() override { return new Member<Class>(mObjectPtr, mFuncPtr); }
		};
    
		  /** CallbackX<...>::Static should be used when setting up a callback to a static or free function **/
		class Static : public Base
		{
			typedef void (*FuncPtr)(Arg1, Arg2, Arg3, Arg4);
			FuncPtr mFuncPtr; 

		public:
			Static(FuncPtr FuncPtr) : mFuncPtr(FuncPtr) { }
              
			void operator()(Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4) override
			{
				mFuncPtr(arg1, arg2, arg3, arg4);
			}

			Base* Clone() override { return new Static(mFuncPtr); }
		};    

	public:    
		Callback() { mPtr = nullptr; }
		Callback(Base* ptr) { mPtr = ptr; }
		Callback(const Callback& cp) { throw NotSupportedException("Cannot copy this object."); }
		Callback(Callback&& mv) { mPtr = mv.mPtr; mv.mPtr = nullptr; }
		~Callback() { if (mPtr != nullptr) delete mPtr; }
		Callback& operator=(Base* aPtr) { if (mPtr != nullptr) delete mPtr; mPtr = aPtr; return *this; }
		Callback& operator=(const Callback& cp) { throw NotSupportedException("Cannot copy this object."); }
		Callback& operator=(Callback&& mv) { if (mPtr != nullptr) delete mPtr; mPtr = mv.mPtr; mv.mPtr = nullptr; return *this; }
		void operator()(Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4) { (*mPtr)(arg1, arg2, arg3, arg4); }
		bool IsPresent() { return mPtr != nullptr; }
	};

	template <class Arg1, class Arg2, class Arg3>
	class Callback<Arg1, Arg2, Arg3, void>
	{
	private:	

		class Base
		{
		public:
			virtual void operator()(Arg1, Arg2, Arg3) = 0;
			virtual Base* Clone() = 0;
			virtual ~Base() { }
		};

		Base*	mPtr;
    
	public:  

		/** CallbackX<...>::Member should be used when setting up a callback to a member function **/
		template <typename Class>
		class Member : public Base
		{		
			typedef void (Class::*FuncPtr)(Arg1, Arg2, Arg3);

			Class*		mObjectPtr;		// Pointer to the object we are delegating to.
			FuncPtr		mFuncPtr;		// Address of the function on the delegate object.

		public:
			Member(Class* ObjectPtr, FuncPtr FuncPtr) : mObjectPtr(ObjectPtr), mFuncPtr(FuncPtr) { }
     
			void operator()(Arg1 arg1, Arg2 arg2, Arg3 arg3) override { (mObjectPtr->*mFuncPtr)(arg1, arg2, arg3); }
			Base* Clone() override { return new Member<Class>(mObjectPtr, mFuncPtr); }
		};
    
		  /** CallbackX<...>::Static should be used when setting up a callback to a static or free function **/
		class Static : public Base
		{
			typedef void (*FuncPtr)(Arg1, Arg2, Arg3);
			FuncPtr mFuncPtr; 

		public:
			Static(FuncPtr FuncPtr) : mFuncPtr(FuncPtr) { }
              
			void operator()(Arg1 arg1, Arg2 arg2, Arg3 arg3) override { mFuncPtr(arg1, arg2, arg3); }
			Base* Clone() override { return new Static(mFuncPtr); }
		};    

	public:    
		Callback() { mPtr = nullptr; }
		Callback(Base* ptr) { mPtr = ptr; }
		Callback(const Callback& cp) { throw NotSupportedException("Cannot copy this object."); }
		Callback(Callback&& mv) { mPtr = mv.mPtr; mv.mPtr = nullptr; }
		~Callback() { if (mPtr != nullptr) delete mPtr; }
		Callback& operator=(Base* aPtr) { if (mPtr != nullptr) delete mPtr; mPtr = aPtr; return *this; }    	
		Callback& operator=(const Callback& cp) { throw NotSupportedException("Cannot copy this object."); }
		Callback& operator=(Callback&& mv) { if (mPtr != nullptr) delete mPtr; mPtr = mv.mPtr; mv.mPtr = nullptr; return *this; }
		void operator()(Arg1 arg1, Arg2 arg2, Arg3 arg3) { (*mPtr)(arg1, arg2, arg3); }
		bool IsPresent() { return mPtr != nullptr; }
	};

	template <class Arg1, class Arg2>
	class Callback<Arg1, Arg2, void, void>
	{
	private:	

		class Base
		{
		public:
			virtual void operator()(Arg1, Arg2) = 0;
			virtual Base* Clone() = 0;
			virtual ~Base() { }
		};

		Base*	mPtr;
    
	public:  

		/** CallbackX<...>::Member should be used when setting up a callback to a member function **/
		template <typename Class>
		class Member : public Base
		{		
			typedef void (Class::*FuncPtr)(Arg1, Arg2);

			Class*		mObjectPtr;		// Pointer to the object we are delegating to.
			FuncPtr		mFuncPtr;		// Address of the function on the delegate object.

		public:
			Member(Class* ObjectPtr, FuncPtr FuncPtr) : mObjectPtr(ObjectPtr), mFuncPtr(FuncPtr) { }
			Member(const Member& cp) : mObjectPtr(cp.mObjectPtr), mFuncPtr(cp.mFuncPtr) { }
     
			void operator()(Arg1 arg1, Arg2 arg2) override
			{
				(mObjectPtr->*mFuncPtr)(arg1, arg2);
			}

			Base* Clone() override { return new Member<Class>(mObjectPtr, mFuncPtr); }
		};
    
		  /** CallbackX<...>::Static should be used when setting up a callback to a static or free function **/
		class Static : public Base
		{
			typedef void (*FuncPtr)(Arg1, Arg2);
			FuncPtr mFuncPtr; 

		public:
			Static(FuncPtr FuncPtr) : mFuncPtr(FuncPtr) { }
              
			void operator()(Arg1 arg1, Arg2 arg2) override
			{
				mFuncPtr(arg1, arg2);
			}

			Base* Clone() override { return new Static(mFuncPtr); }
		};    

	public:    
		Callback() { mPtr = nullptr; }
		Callback(Base* ptr) { mPtr = ptr; }
		Callback(const Callback& cp) { throw NotSupportedException("Cannot copy this object."); }
		Callback(Callback&& mv) { mPtr = mv.mPtr; mv.mPtr = nullptr; }
		~Callback() { if (mPtr != nullptr) delete mPtr; }		
		Callback& operator=(Base* aPtr) { if (mPtr != nullptr) delete mPtr; mPtr = aPtr; return *this; }    	
		Callback& operator=(const Callback& cp) { throw NotSupportedException("Cannot copy this object."); }
		Callback& operator=(Callback&& mv) { if (mPtr != nullptr) delete mPtr; mPtr = mv.mPtr; mv.mPtr = nullptr; return *this; }
		void operator()(Arg1 arg1, Arg2 arg2) { (*mPtr)(arg1, arg2); }
		bool IsPresent() { return mPtr != nullptr; }
	};

	template <class Arg1>
	class Callback<Arg1, void, void, void>
	{
	private:	

		class Base
		{
		public:
			virtual void operator()(Arg1) = 0;
			virtual Base* Clone() = 0;
			virtual ~Base() { }
		};

		Base*	mPtr;
    
	public:  

		/** CallbackX<...>::Member should be used when setting up a callback to a member function **/
		template <typename Class>
		class Member : public Base
		{		
			typedef void (Class::*FuncPtr)(Arg1);

			Class*		mObjectPtr;		// Pointer to the object we are delegating to.
			FuncPtr		mFuncPtr;		// Address of the function on the delegate object.

		public:
			Member(Class* ObjectPtr, FuncPtr FuncPtr) : mObjectPtr(ObjectPtr), mFuncPtr(FuncPtr) { }
     
			void operator()(Arg1 arg1) override
			{
				(mObjectPtr->*mFuncPtr)(arg1);
			}

			Base* Clone() override { return new Member<Class>(mObjectPtr, mFuncPtr); }
		};
    
		  /** CallbackX<...>::Static should be used when setting up a callback to a static or free function **/
		class Static : public Base
		{
			typedef void (*FuncPtr)(Arg1);
			FuncPtr mFuncPtr; 

		public:
			Static(FuncPtr FuncPtr) : mFuncPtr(FuncPtr) { }
              
			void operator()(Arg1 arg1) override
			{
				mFuncPtr(arg1);
			}

			Base* Clone() override { return new Static(mFuncPtr); }
		};    

	public:    
		Callback() { mPtr = nullptr; }
		Callback(Base* ptr) { mPtr = ptr; }
		Callback(const Callback& cp) { throw NotSupportedException("Cannot copy this object."); }
		Callback(Callback&& mv) { mPtr = mv.mPtr; mv.mPtr = nullptr; }
		~Callback() { if (mPtr != nullptr) delete mPtr; }
		Callback& operator=(Base* aPtr) { if (mPtr != nullptr) delete mPtr; mPtr = aPtr; return *this; }
		Callback& operator=(const Callback& cp) { throw NotSupportedException("Cannot copy this object."); }
		Callback& operator=(Callback&& mv) { if (mPtr != nullptr) delete mPtr; mPtr = mv.mPtr; mv.mPtr = nullptr; return *this; }
		void operator()(Arg1 arg1) { (*mPtr)(arg1); }
		bool IsPresent() { return mPtr != nullptr; }
	};

	template <>
	class Callback<void, void, void, void>
	{
	private:	

		class Base
		{
		public:
			virtual void operator()(void) = 0;
			virtual Base* Clone() = 0;
			virtual ~Base() { }
		};

		Base*	mPtr;
    
	public:  

		/** CallbackX<...>::Member should be used when setting up a callback to a member function **/
		template <typename Class>
		class Member : public Base
		{		
			typedef void (Class::*FuncPtr)(void);

			Class*		mObjectPtr;		// Pointer to the object we are delegating to.
			FuncPtr		mFuncPtr;		// Address of the function on the delegate object.

		public:
			Member(Class* ObjectPtr, FuncPtr FuncPtr) : mObjectPtr(ObjectPtr), mFuncPtr(FuncPtr) { }
     
			void operator()(void) override
			{
				(mObjectPtr->*mFuncPtr)();
			}

			Base* Clone() override { return new Member<Class>(mObjectPtr, mFuncPtr); }
		};
    
		  /** CallbackX<...>::Static should be used when setting up a callback to a static or free function **/
		class Static : public Base
		{
			typedef void (*FuncPtr)(void);
			FuncPtr mFuncPtr; 

		public:
			Static(FuncPtr FuncPtr) : mFuncPtr(FuncPtr) { }
              
			void operator()(void) override
			{
				mFuncPtr();
			}

			Base* Clone() override { return new Static(mFuncPtr); }
		};

	public:    
		Callback() { mPtr = nullptr; }
		Callback(Base* ptr) { mPtr = ptr; }
		Callback(const Callback& cp) { throw NotSupportedException("Cannot copy this object."); }
		Callback(Callback&& mv) { mPtr = mv.mPtr; mv.mPtr = nullptr; }
		~Callback() { if (mPtr != nullptr) delete mPtr; }
		Callback& operator=(Base* aPtr) { if (mPtr != nullptr) delete mPtr; mPtr = aPtr; return *this; }    	
		Callback& operator=(const Callback& cp) { throw NotSupportedException("Cannot copy this object."); }
		Callback& operator=(Callback&& mv) { if (mPtr != nullptr) delete mPtr; mPtr = mv.mPtr; mv.mPtr = nullptr; return *this; }
		void operator()(void) { (*mPtr)(); }
		bool IsPresent() { return mPtr != nullptr; }
	};
}

#endif

//	End of Callback.h

