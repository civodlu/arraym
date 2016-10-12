/*
 * Numerical learning library
 * http://nll.googlecode.com/
 *
 * Copyright (c) 2009-2012, Ludovic Sibille
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Ludovic Sibille nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY LUDOVIC SIBILLE ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NLL_TIMER_H_
# define NLL_TIMER_H_

namespace nll
{
namespace core
{
   /**
    @ingroup core
    @brief Define a simple Timer class.
   */
   class INFRA_API Timer
   {
   public:
      /**
       @brief Instanciate the timer and start it
       */
      Timer()
      {
         start();
         _end = _start;
      }

      /**
       @brief restart the timer
       */
      void start()
      {
         _start = std::chrono::high_resolution_clock::now();
      }

      /**
       @brief end the timer, return the time in seconds spent
       */
      void end()
      {
         _end = std::chrono::high_resolution_clock::now();
      }

      /**
       @brief get the current time since the begining, return the time in seconds spent.
       */
      f32 getElapsedTime() const
      {
         auto c = std::chrono::high_resolution_clock::now();
         return toSeconds( _start, c );
      }

      /**
       @brief return the time in seconds spent since between starting and ending the timer. The timer needs to be ended before calling it.
       */
      f32 getTime() const
      {
         return toSeconds( _start, _end );
      }

   private:
      static f32 toSeconds( const std::chrono::high_resolution_clock::time_point& start,
                            const std::chrono::high_resolution_clock::time_point& end )
      {
         const auto c = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
         return static_cast<f32>( c / 1000.0f);
      }

   private:
      std::chrono::high_resolution_clock::time_point _start;
      std::chrono::high_resolution_clock::time_point _end;
   };
}
}

#endif
