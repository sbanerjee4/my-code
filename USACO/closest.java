import java.util.*;
import java.io.*;
import java.lang.*;

public class closest
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(System.in);
      int k = in.nextInt(), m = in.nextInt(), n = in.nextInt();
      Obj[] a = new Obj[k + m];
      for(int i = 0; i < k; i++)
         a[i] = new Obj(in.nextLong(), in.nextLong());
      for(int i = k; i < m + k; i++)
         a[i] = new Obj(in.nextLong());
      in.close();
     
      Arrays.sort(a, new ComparatorObj());
      PriorityQueue<Long> vals = new PriorityQueue<>();
      long gap = Long.MAX_VALUE;
      long count = 0;
      long sum = 0;
      
      if(n != 0)
      {
         for(int i = 0; i < m + k - 1; i++)
         {
            if(i == 0 && a[i].cow)
            {
               count++;
               sum += a[0].t;
               vals.add(a[0].t);
            }
            if(!a[i].cow && !a[i + 1].cow)
            {
               gap = Long.MAX_VALUE;
               continue;
            }
            else if(!a[i].cow && a[i + 1].cow)
            {
               gap = a[i + 1].pos - a[i].pos;
               if(count + 1 <= n)
               {
                  count++;
                  vals.add(a[i + 1].t);
                  sum += a[i + 1].t;
               }
               else
               {
                  if(sum - (long)vals.toArray()[0] + a[i + 1].t > sum)
                  {
                     sum = sum - (long)vals.toArray()[0] + a[i + 1].t;
                     vals.remove((long)vals.toArray()[0]);
                     vals.add(a[i + 1].t);
                  }
               }
            }
            else if(a[i].cow && !a[i + 1].cow)
            {
               gap = Long.MAX_VALUE;
               continue;
            }
            else
            {
               if(a[i + 1].pos - a[i].pos <= gap)
               {
                  vals.add(a[i + 1].t);
                  sum += a[i + 1].t;
               }
            }
         }
      }
          
      System.out.println(sum);
   }
   
   static class Obj
   {
      public long pos;
      public long t;
      public boolean cow;
      public Obj(long a, long b)
      {
         pos = a;
         t = b;
         cow = true;
      }
      
      public Obj(long a)
      {
         pos = a;
         t = 0;
         cow = false;
      }
   }
   
   static class ComparatorObj implements Comparator<Obj>
   {
      public int compare(Obj a, Obj b)
      {
         return Long.compare(a.pos, b.pos);
      }
   }
}